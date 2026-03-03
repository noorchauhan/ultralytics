from __future__ import annotations

import logging
import math

import torch
import torch.nn.functional as F

from ultralytics.utils.loss import DFLoss, v8DetectionLoss

LOGGER = logging.getLogger(__name__)


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

    def loss(self, preds: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """Calculate stereo 3D detection loss: det losses + aux 3D losses.

        Args:
            preds: Dict with boxes, scores, feats, lr_distance, depth, dimensions, orientation.
            batch: Batch dict with img, batch_idx, cls, bboxes, aux_targets.
        """
        # Separate aux preds from detection preds
        aux_keys = {"lr_distance", "depth", "depth_bins", "dimensions", "orientation"}
        aux_preds = {k: v for k, v in preds.items() if k in aux_keys}

        loss = torch.zeros(6, device=self.device)  # box, cls, lr_dist, depth, dims, orient

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

        # Diagnostic logging every 50 steps
        if not hasattr(self, "_step"):
            self._step = 0
        self._step += 1
        if self._step % 50 == 1:
            self._log_diagnostics(preds, aux_preds, batch, fg_mask, target_gt_idx)

        batch_size = preds["boxes"].shape[0]
        return loss * batch_size, loss.detach()

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
