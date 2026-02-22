from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from ultralytics.utils.loss import DFLoss, v8DetectionLoss


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
    ):
        super().__init__(model, tal_topk=tal_topk)
        self.aux_w = loss_weights or {}
        self.use_bbox_loss = use_bbox_loss

        # Depth bin classification (DFL-style)
        from ultralytics.models.yolo.stereo3ddet.head_yolo11 import DEPTH_BINS, DEPTH_MAX, DEPTH_MIN

        self.depth_dfl_loss = DFLoss(reg_max=DEPTH_BINS)
        self.depth_log_min = math.log(DEPTH_MIN)
        self.depth_log_range = math.log(DEPTH_MAX) - math.log(DEPTH_MIN)

    def _aux_loss(
        self,
        pred_map: torch.Tensor,
        aux_gt: torch.Tensor,
        gt_idx: torch.Tensor,
        fg_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Compute auxiliary loss on positives using gathered GT via target_gt_idx.

        Args:
            pred_map: [B, C, HW_total] — 3D flattened aux predictions.
            aux_gt: [B, max_n, C] — padded per-image GT.
            gt_idx: [B, HW_total] — assignment indices from TAL.
            fg_mask: [B, HW_total] — boolean foreground mask.
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

        return F.smooth_l1_loss(pred_pos, tgt_pos, reduction="mean")

    def _compute_aux_losses(
        self,
        aux_preds: dict[str, torch.Tensor],
        batch: dict[str, torch.Tensor],
        target_gt_idx: torch.Tensor,
        fg_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Compute auxiliary losses for all 3D heads."""
        aux_losses: dict[str, torch.Tensor] = {}
        aux_targets = batch.get("aux_targets", {})

        if not isinstance(aux_targets, dict) or not aux_targets:
            return aux_losses

        for k in ("lr_distance", "depth", "dimensions", "orientation"):
            if k not in aux_targets:
                continue
            aux_gt = aux_targets[k].to(self.device)
            if k == "depth" and "depth_bins" in aux_preds:
                aux_losses[k] = self._depth_bin_loss(aux_preds["depth_bins"], aux_gt, target_gt_idx, fg_mask)
            elif k in aux_preds:
                aux_losses[k] = self._aux_loss(aux_preds[k], aux_gt, target_gt_idx, fg_mask)

        return aux_losses

    def _depth_bin_loss(
        self,
        pred_bins: torch.Tensor,
        aux_gt: torch.Tensor,
        gt_idx: torch.Tensor,
        fg_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Compute DFL-style depth bin classification loss.

        Args:
            pred_bins: [B, n_bins, HW_total] — raw logits from depth branch.
            aux_gt: [B, max_n, 1] — log-depth GT values.
            gt_idx: [B, HW_total] — TAL assignment indices.
            fg_mask: [B, HW_total] — boolean foreground mask.
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

        return self.depth_dfl_loss(pred_fg, tgt_fg.unsqueeze(-1)).mean()

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

        batch_size = preds["boxes"].shape[0]
        return loss * batch_size, loss.detach()
