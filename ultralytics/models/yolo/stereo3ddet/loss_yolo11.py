from __future__ import annotations

import logging
import math

import torch
import torch.nn.functional as F

from ultralytics.utils.loss import DFLoss, v8DetectionLoss

LOGGER = logging.getLogger(__name__)


def _ssim(x: torch.Tensor, y: torch.Tensor, C1: float = 1e-4, C2: float = 9e-4) -> torch.Tensor:
    """Compute per-pixel SSIM between two images (assumed [0,1] range).

    Returns [B, 1, H, W] SSIM map in [-1, 1] (1 = identical).
    """
    pad = 1  # kernel_size=3, pad=1
    mu_x = F.avg_pool2d(x, 3, 1, pad)
    mu_y = F.avg_pool2d(y, 3, 1, pad)
    sigma_x2 = F.avg_pool2d(x * x, 3, 1, pad) - mu_x * mu_x
    sigma_y2 = F.avg_pool2d(y * y, 3, 1, pad) - mu_y * mu_y
    sigma_xy = F.avg_pool2d(x * y, 3, 1, pad) - mu_x * mu_y
    ssim_map = ((2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)) / (
        (mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x2 + sigma_y2 + C2)
    )
    return ssim_map.mean(dim=1, keepdim=True)  # average over RGB → [B, 1, H, W]


class Stereo3DDetLossYOLO11(v8DetectionLoss):
    """Multi-scale loss for stereo 3D detection using YOLO-style bbox assignment.

    Overrides loss() to add auxiliary 3D losses (lr_distance, depth, dimensions,
    orientation) on top of the standard detection losses (box, cls, dfl).

    Expected preds dict keys (from head's forward_head):
        - boxes, scores, feats: standard Detect outputs
        - lr_distance, depth, dimensions, orientation: aux branch outputs [B, C, HW_total]

    Expected batch keys:
        - img, batch_idx, cls, bboxes: standard YOLO detection targets
        - aux_targets: dict[str, Tensor] each [B, max_n, C] in pixel units
    """

    def __init__(
        self,
        model,
        tal_topk: int = 10,
        loss_weights: dict[str, float] | None = None,
        use_bbox_loss: bool = True,
        cls_label_smoothing: float = 0.0,
    ):
        super().__init__(model, tal_topk=tal_topk)
        self.aux_w = loss_weights or {}
        self.use_bbox_loss = use_bbox_loss
        self.cls_label_smoothing = cls_label_smoothing
        self.photo_w = float(self.aux_w.get("photometric", 0.0))
        self.smooth_w = float(self.aux_w.get("smoothness", 0.001))

        # Depth bin classification (DFL-style)
        from ultralytics.models.yolo.stereo3ddet.head_yolo11 import DEPTH_BINS, DEPTH_MAX, DEPTH_MIN

        self.depth_dfl_loss = DFLoss(reg_max=DEPTH_BINS)
        self.depth_log_min = math.log(DEPTH_MIN)
        self.depth_log_range = math.log(DEPTH_MAX) - math.log(DEPTH_MIN)

        # Pseudo-label curriculum: set by trainer callback each epoch
        self.epoch_frac = 0.0  # 0.0 = start, 1.0 = end of training
        # Aux loss weight multiplier for pseudo-labels (stereo-pseudo gets full, mono gets reduced)
        self._pseudo_stereo_w = 0.5  # stereo-matched pseudo: 50% aux loss weight
        self._pseudo_mono_w = 0.2  # mono-only pseudo: 20% aux loss weight
        self._pseudo_cutoff = 0.9  # phase out all pseudo-labels after this fraction

    def _pseudo_aux_weights(self, is_pseudo_fg: torch.Tensor) -> torch.Tensor:
        """Compute per-anchor aux loss weight based on pseudo-label flag and epoch.

        Args:
            is_pseudo_fg: [npos, 1] — 0=real, 1=stereo-pseudo, 2=mono-pseudo.

        Returns:
            [npos, 1] weight tensor: 1.0 for real, reduced for pseudo, 0 after cutoff.
        """
        w = torch.ones_like(is_pseudo_fg)

        # Schedule: linear decay in final phase, then hard cutoff
        if self.epoch_frac >= self._pseudo_cutoff:
            # After cutoff: pseudo labels contribute 0 to aux losses
            pseudo_mask = is_pseudo_fg > 0
            w[pseudo_mask] = 0.0
        else:
            # Before cutoff: reduced weight for pseudo labels
            # Linear ramp-down: full weight at epoch 0, half at cutoff
            schedule = 1.0 - 0.5 * (self.epoch_frac / self._pseudo_cutoff)
            w[is_pseudo_fg == 1] = self._pseudo_stereo_w * schedule
            w[is_pseudo_fg == 2] = self._pseudo_mono_w * schedule

        return w

    def _aux_loss(
        self,
        pred_map: torch.Tensor,
        aux_gt: torch.Tensor,
        gt_idx: torch.Tensor,
        fg_mask: torch.Tensor,
        aux_weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute auxiliary loss on positives using gathered GT via target_gt_idx.

        Args:
            pred_map: [B, C, HW_total] — 3D flattened aux predictions.
            aux_gt: [B, max_n, C] — padded per-image GT.
            gt_idx: [B, HW_total] — assignment indices from TAL.
            fg_mask: [B, HW_total] — boolean foreground mask.
            aux_weights: [npos, 1] — per-anchor weight (pseudo-label curriculum).
        """
        bs, c, n = pred_map.shape
        pred_flat = pred_map.permute(0, 2, 1)  # [B, HW_total, C]

        if aux_gt.shape[1] == 0:
            return pred_map.sum() * 0.0

        if gt_idx.dtype != torch.int64:
            gt_idx = gt_idx.to(torch.int64)
        gathered = aux_gt.gather(1, gt_idx.unsqueeze(-1).expand(-1, -1, c))  # [B, HW_total, C]

        pred_pos = pred_flat[fg_mask]  # [npos, C]
        tgt_pos = gathered[fg_mask]  # [npos, C]

        if pred_pos.numel() == 0:
            return pred_map.sum() * 0.0

        if aux_weights is not None:
            # Weighted mean: per-anchor loss × weight, normalized by weight sum
            raw = F.smooth_l1_loss(pred_pos, tgt_pos, reduction="none")  # [npos, C]
            return (raw.mean(-1, keepdim=True) * aux_weights).sum() / aux_weights.sum().clamp(min=1.0)

        return F.smooth_l1_loss(pred_pos, tgt_pos, reduction="mean")

    def _compute_aux_losses(
        self,
        aux_preds: dict[str, torch.Tensor],
        batch: dict[str, torch.Tensor],
        target_gt_idx: torch.Tensor,
        fg_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Compute auxiliary losses for all 3D heads with pseudo-label weighting."""
        aux_losses: dict[str, torch.Tensor] = {}
        aux_targets = batch.get("aux_targets", {})

        if not isinstance(aux_targets, dict) or not aux_targets:
            return aux_losses

        # Compute per-anchor pseudo-label weights for aux losses
        aux_weights = None
        is_pseudo_gt = aux_targets.get("is_pseudo")
        if is_pseudo_gt is not None and is_pseudo_gt.shape[1] > 0:
            is_pseudo_gt = is_pseudo_gt.to(self.device)
            if target_gt_idx.dtype != torch.int64:
                target_gt_idx = target_gt_idx.to(torch.int64)
            gathered = is_pseudo_gt.gather(1, target_gt_idx.unsqueeze(-1))  # [B, HW, 1]
            is_pseudo_fg = gathered[fg_mask]  # [npos, 1]
            if is_pseudo_fg.any():
                aux_weights = self._pseudo_aux_weights(is_pseudo_fg)

        for k in ("lr_distance", "depth", "dimensions", "orientation"):
            if k not in aux_targets:
                continue
            aux_gt = aux_targets[k].to(self.device)
            if k == "depth" and "depth_bins" in aux_preds:
                aux_losses[k] = self._depth_bin_loss(
                    aux_preds["depth_bins"], aux_gt, target_gt_idx, fg_mask, aux_weights,
                )
            elif k in aux_preds:
                aux_losses[k] = self._aux_loss(aux_preds[k], aux_gt, target_gt_idx, fg_mask, aux_weights)

        return aux_losses

    def _depth_bin_loss(
        self,
        pred_bins: torch.Tensor,
        aux_gt: torch.Tensor,
        gt_idx: torch.Tensor,
        fg_mask: torch.Tensor,
        aux_weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute DFL-style depth bin classification loss.

        Args:
            pred_bins: [B, n_bins, HW_total] — raw logits from depth branch.
            aux_gt: [B, max_n, 1] — log-depth GT values.
            gt_idx: [B, HW_total] — TAL assignment indices.
            fg_mask: [B, HW_total] — boolean foreground mask.
            aux_weights: [npos, 1] — per-anchor weight (pseudo-label curriculum).
        """
        n_bins = pred_bins.shape[1]
        if aux_gt.shape[1] == 0 or not fg_mask.any():
            return pred_bins.sum() * 0.0

        # Gather GT log-depth per anchor
        if gt_idx.dtype != torch.int64:
            gt_idx = gt_idx.to(torch.int64)
        gathered = aux_gt.gather(1, gt_idx.unsqueeze(-1))  # [B, HW_total, 1]

        # Convert log-depth → fractional bin index
        bin_idx = (gathered - self.depth_log_min) / self.depth_log_range * (n_bins - 1)

        # Select foreground
        pred_fg = pred_bins.permute(0, 2, 1)[fg_mask]  # [npos, n_bins]
        tgt_fg = bin_idx.squeeze(-1)[fg_mask]  # [npos]

        if pred_fg.numel() == 0:
            return pred_bins.sum() * 0.0

        raw = self.depth_dfl_loss(pred_fg, tgt_fg.unsqueeze(-1))  # [npos, 1]
        if aux_weights is not None:
            return (raw * aux_weights).sum() / aux_weights.sum().clamp(min=1.0)
        return raw.mean()

    def _photometric_loss(
        self,
        dense_log_depth: torch.Tensor,
        batch: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Self-supervised photometric reconstruction loss.

        Warps the right image to the left viewpoint using predicted dense depth,
        then penalizes SSIM + L1 difference. Provides dense depth supervision
        that bypasses TAL assignments — critical for nc=1 training.

        Args:
            dense_log_depth: [B, 1, Hd, Wd] dense log-depth at feature stride.
            batch: Batch dict with img [B, 6, H, W] and calib list.

        Returns:
            Scalar photometric + smoothness loss.
        """
        B, _, Hd, Wd = dense_log_depth.shape
        device = dense_log_depth.device

        # Split stereo pair and downsample to feature resolution
        img = batch["img"]
        stride = img.shape[2] / Hd  # dynamic stride (4 for tap features, 8 for P3)
        left = F.interpolate(img[:, :3], size=(Hd, Wd), mode="bilinear", align_corners=False)
        right = F.interpolate(img[:, 3:], size=(Hd, Wd), mode="bilinear", align_corners=False)

        # Build per-image fx and baseline tensors [B, 1, 1, 1]
        calibs = batch.get("calib", [{}] * B)
        fx_vals = torch.tensor([c.get("fx", 721.0) for c in calibs], device=device, dtype=left.dtype)
        bl_vals = torch.tensor([c.get("baseline", 0.54) for c in calibs], device=device, dtype=left.dtype)
        fx_scaled = (fx_vals / stride).view(B, 1, 1, 1)  # scale focal length to feature resolution
        baseline = bl_vals.view(B, 1, 1, 1)

        # Log-depth → disparity in feature pixels
        depth_m = dense_log_depth.exp().clamp(min=0.1, max=200.0)
        disparity = fx_scaled * baseline / depth_m  # [B, 1, Hd, Wd]

        # Build sampling grid: for left pixel (u,v), source is right pixel (u - disp, v)
        u = torch.linspace(-1, 1, Wd, device=device, dtype=left.dtype)
        v = torch.linspace(-1, 1, Hd, device=device, dtype=left.dtype)
        grid_v, grid_u = torch.meshgrid(v, u, indexing="ij")  # [Hd, Wd]
        base_u = grid_u.unsqueeze(0).expand(B, -1, -1)  # [B, Hd, Wd]
        base_v = grid_v.unsqueeze(0).expand(B, -1, -1)

        # Convert disparity from pixels to normalized [-1,1] coordinates
        disp_norm = disparity.squeeze(1) * 2.0 / (Wd - 1)  # [B, Hd, Wd]
        sample_u = base_u - disp_norm
        grid = torch.stack([sample_u, base_v], dim=-1)  # [B, Hd, Wd, 2]

        # Warp right image to left viewpoint
        warped = F.grid_sample(right, grid, mode="bilinear", padding_mode="zeros", align_corners=True)

        # Photometric error: alpha * SSIM + (1-alpha) * L1
        alpha = 0.85
        l1_err = (warped - left).abs().mean(dim=1, keepdim=True)  # [B, 1, Hd, Wd]
        ssim_err = (1.0 - _ssim(warped, left)) / 2.0  # [B, 1, Hd, Wd]
        photo_err = alpha * ssim_err + (1 - alpha) * l1_err

        # Identity error (unwarped right vs left) for auto-masking
        id_l1 = (right - left).abs().mean(dim=1, keepdim=True)
        id_ssim = (1.0 - _ssim(right, left)) / 2.0
        id_err = alpha * id_ssim + (1 - alpha) * id_l1

        # Auto-mask: only penalize where warping improves over identity
        auto_mask = (photo_err < id_err).float()

        # Validity mask: exclude pixels where sampling falls outside image
        valid = ((grid[..., 0] > -1) & (grid[..., 0] < 1)).unsqueeze(1).float()  # [B, 1, Hd, Wd]
        mask = valid * auto_mask

        n_valid = mask.sum().clamp(min=1.0)
        loss_photo = (photo_err * mask).sum() / n_valid

        # Edge-aware smoothness on normalized depth
        if self.smooth_w > 0:
            norm_depth = dense_log_depth / (dense_log_depth.detach().mean(dim=[2, 3], keepdim=True).abs() + 1e-7)
            dx_d = (norm_depth[:, :, :, :-1] - norm_depth[:, :, :, 1:]).abs()
            dy_d = (norm_depth[:, :, :-1, :] - norm_depth[:, :, 1:, :]).abs()
            dx_i = left[:, :, :, :-1].sub(left[:, :, :, 1:]).abs().mean(1, keepdim=True)
            dy_i = left[:, :, :-1, :].sub(left[:, :, 1:, :]).abs().mean(1, keepdim=True)
            loss_smooth = (dx_d * (-dx_i).exp()).mean() + (dy_d * (-dy_i).exp()).mean()
            loss_photo = loss_photo + self.smooth_w * loss_smooth

        return loss_photo

    def loss(self, preds: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """Calculate stereo 3D detection loss: det losses + aux 3D losses.

        Args:
            preds: Dict with boxes, scores, feats, lr_distance, depth, dimensions, orientation.
            batch: Batch dict with img, batch_idx, cls, bboxes, aux_targets.
        """
        # Separate aux preds from detection preds
        aux_keys = {"lr_distance", "depth", "depth_bins", "dimensions", "orientation", "dense_depth"}
        aux_preds = {k: v for k, v in preds.items() if k in aux_keys}

        loss = torch.zeros(8, device=self.device)  # box, cls, lr_dist, depth, dims, orient, divers, photo

        # Get detection losses + TAL assignment results
        (fg_mask, target_gt_idx, target_bboxes, anchor_points, stride_tensor), det_loss, _ = (
            self.get_assigned_targets_and_loss(preds, batch)
        )

        if self.use_bbox_loss:
            loss[0] = det_loss[0]  # box (already scaled by hyp.box)
        loss[1] = det_loss[1]  # cls (already scaled by hyp.cls)
        # det_loss[2] is dfl, which is 0 since reg_max=1

        # Aux losses
        aux_losses = self._compute_aux_losses(aux_preds, batch, target_gt_idx, fg_mask)
        for i, k in enumerate(["lr_distance", "depth", "dimensions", "orientation"], 2):
            if k in aux_losses:
                loss[i] = aux_losses[k] * float(self.aux_w.get(k, 1.0))

        # Feature diversity loss (decorrelation on P3 features)
        diversity_w = float(self.aux_w.get("diversity", 0.0))
        if diversity_w > 0 and "p3_features" in preds:
            loss[6] = self._feature_diversity_loss(preds["p3_features"]) * diversity_w

        # Self-supervised photometric reconstruction loss
        if self.photo_w > 0 and "dense_depth" in preds:
            loss[7] = self._photometric_loss(preds["dense_depth"], batch) * self.photo_w

        # Diagnostic logging every 50 steps
        if not hasattr(self, "_step"):
            self._step = 0
        self._step += 1
        if self._step % 50 == 1:
            self._log_diagnostics(preds, aux_preds, batch, fg_mask, target_gt_idx)

        batch_size = preds["boxes"].shape[0]
        return loss * batch_size, loss.detach()

    def _feature_diversity_loss(self, features: torch.Tensor) -> torch.Tensor:
        """Decorrelation loss on P3 feature channels (Barlow Twins style).

        Penalizes correlated channels to maintain feature diversity and prevent
        the effective rank collapse that causes nc=1 depth failure.

        Args:
            features: [B, C, H, W] P3 feature map from neck.

        Returns:
            Scalar loss: mean squared off-diagonal correlation (0=decorrelated, 1=identical).
        """
        B, C, H, W = features.shape
        x = features.float().permute(0, 2, 3, 1).reshape(-1, C)  # [N, C] where N=B*H*W
        x = x - x.mean(dim=0, keepdim=True)
        # Correlation matrix
        cov = (x.T @ x) / (x.shape[0] - 1)  # [C, C]
        std = cov.diag().sqrt().clamp(min=1e-5)
        corr = cov / (std.unsqueeze(0) * std.unsqueeze(1))
        # Off-diagonal mean squared correlation
        mask = ~torch.eye(C, dtype=torch.bool, device=corr.device)
        return corr[mask].pow(2).mean()

    @torch.no_grad()
    def _log_diagnostics(self, preds, aux_preds, batch, fg_mask, target_gt_idx):
        """Log depth prediction and feature diagnostics."""
        n_fg = fg_mask.sum().item()
        if n_fg == 0:
            return

        # 1. Depth prediction distribution at fg anchors
        if "depth" in aux_preds:
            depth_pred = aux_preds["depth"]  # [B, 1, HW] log-depth
            pred_fg = depth_pred.permute(0, 2, 1)[fg_mask]  # [npos, 1]
            pred_m = pred_fg.exp()  # meters

            # GT depth
            aux_targets = batch.get("aux_targets", {})
            if "depth" in aux_targets:
                gt = aux_targets["depth"].to(depth_pred.device)
                if gt.shape[1] > 0:
                    gt_fg = gt.gather(1, target_gt_idx.unsqueeze(-1).long())[fg_mask]
                    gt_m = gt_fg.exp()

                    err = (pred_m - gt_m).abs()
                    LOGGER.info(
                        "DEPTH step=%d n_fg=%d | pred: mean=%.1fm std=%.1fm min=%.1fm max=%.1fm | "
                        "gt: mean=%.1fm std=%.1fm | err: mean=%.1fm median=%.1fm",
                        self._step, n_fg,
                        pred_m.mean(), pred_m.std(), pred_m.min(), pred_m.max(),
                        gt_m.mean(), gt_m.std(),
                        err.mean(), err.median(),
                    )

        # 2. Depth bin logit distribution (pre-softmax)
        if "depth_bins" in aux_preds:
            bins = aux_preds["depth_bins"]  # [B, 16, HW]
            bins_fg = bins.permute(0, 2, 1)[fg_mask]  # [npos, 16]
            probs = bins_fg.softmax(dim=1)
            entropy = -(probs * probs.clamp(min=1e-8).log()).sum(dim=1)  # [npos]
            max_entropy = math.log(bins_fg.shape[1])  # log(16) = 2.77

            # Peak bin: which bin gets highest probability
            peak_bins = probs.argmax(dim=1)  # [npos]

            LOGGER.info(
                "BINS  step=%d | entropy: mean=%.2f/%.2f (%.0f%% of max) | "
                "peak_bin: mean=%.1f std=%.1f | logit_range: [%.1f, %.1f]",
                self._step,
                entropy.mean(), max_entropy, entropy.mean() / max_entropy * 100,
                peak_bins.float().mean(), peak_bins.float().std(),
                bins_fg.min(), bins_fg.max(),
            )

        # 3. Classification confidence at fg anchors
        if "scores" in preds:
            scores = preds["scores"]  # [B, nc, HW]
            scores_fg = scores.permute(0, 2, 1)[fg_mask].sigmoid()  # [npos, nc]
            LOGGER.info(
                "CLS   step=%d | fg_conf: mean=%.3f max=%.3f | nc=%d",
                self._step, scores_fg.mean(), scores_fg.max(), scores.shape[1],
            )

        # 4. Pseudo-label stats
        aux_targets = batch.get("aux_targets", {})
        is_pseudo_gt = aux_targets.get("is_pseudo")
        if is_pseudo_gt is not None and is_pseudo_gt.shape[1] > 0:
            gathered = is_pseudo_gt.to(fg_mask.device).gather(
                1, target_gt_idx.unsqueeze(-1).long(),
            )
            pf = gathered[fg_mask].squeeze(-1)
            n_real = (pf == 0).sum().item()
            n_stereo = (pf == 1).sum().item()
            n_mono = (pf == 2).sum().item()
            if n_stereo + n_mono > 0:
                LOGGER.info(
                    "PSEUDO step=%d | fg: %d real + %d stereo + %d mono | epoch_frac=%.2f",
                    self._step, n_real, n_stereo, n_mono, self.epoch_frac,
                )

        # 5. Feature diversity metrics
        if "p3_features" in preds:
            p3 = preds["p3_features"]
            B, C, H, W = p3.shape
            x = p3.float().permute(0, 2, 3, 1).reshape(-1, C)
            x = x - x.mean(dim=0, keepdim=True)
            # Effective rank via SVD (subsample for speed)
            if x.shape[0] > 4096:
                idx = torch.randperm(x.shape[0], device=x.device)[:4096]
                x_sub = x[idx]
            else:
                x_sub = x
            s = torch.linalg.svdvals(x_sub)
            p_s = s / s.sum()
            eff_rank = (-(p_s * p_s.clamp(min=1e-8).log()).sum()).exp()
            LOGGER.info(
                "FEAT  step=%d | eff_rank=%.1f/%d | top5_sv=[%.1f,%.1f,%.1f,%.1f,%.1f]",
                self._step, eff_rank.item(), C, *s[:5].tolist(),
            )
