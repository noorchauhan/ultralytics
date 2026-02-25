from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from ultralytics.utils.loss import DFLoss, E2ELoss, v8DetectionLoss
from ultralytics.utils.tal import make_anchors

# Reference nc for cls loss normalization — cls BCE sums over B*HW*nc, so fewer
# classes means weaker cls signal, worse TAL assignments, and aux branch starvation.
# Normalizing by nc makes cls loss magnitude (and thus TAL quality) nc-invariant.
_CLS_NC_REF = 3


class Stereo3DDetLossYOLO11(v8DetectionLoss):
    """Multi-scale loss for stereo 3D detection using YOLO-style bbox assignment.

    Overrides loss() to add auxiliary 3D losses (lr_distance, depth, dimensions,
    orientation) on top of the standard detection losses (box, cls, dfl).

    Key design: cls loss is normalized by nc to ensure stable TAL assignments
    regardless of number of classes. Without this, nc=1 produces ~3x weaker cls
    gradients, degrading TAL quality and causing aux branch collapse.

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
        tal_topk2: int | None = None,
    ):
        super().__init__(model, tal_topk=tal_topk, tal_topk2=tal_topk2)

        # Read loss config from model YAML (E2ELoss creates instances with just model+topk args)
        yaml_cfg = getattr(model, "yaml", None) or {}
        training_cfg = yaml_cfg.get("training", {})
        self.aux_w = training_cfg.get("loss_weights", {})
        self.use_bbox_loss = bool(training_cfg.get("use_bbox_loss", True))

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

        Inlines TAL assignment from parent. Cls loss is normalized by nc to keep
        TAL assignment quality stable regardless of number of classes.
        """
        # Separate aux preds from detection preds
        aux_keys = {"lr_distance", "depth", "depth_bins", "dimensions", "orientation"}
        aux_preds = {k: v for k, v in preds.items() if k in aux_keys}

        loss = torch.zeros(6, device=self.device)  # box, cls, lr_dist, depth, dims, orient

        # --- Inline TAL assignment (from parent get_assigned_targets_and_loss) ---
        pred_distri, pred_scores = (
            preds["boxes"].permute(0, 2, 1).contiguous(),
            preds["scores"].permute(0, 2, 1).contiguous(),
        )
        anchor_points, stride_tensor = make_anchors(preds["feats"], self.stride, 0.5)

        dtype = pred_scores.dtype
        batch_size = pred_scores.shape[0]
        imgsz = torch.tensor(preds["feats"][0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]

        targets = torch.cat((batch["batch_idx"].view(-1, 1), batch["cls"].view(-1, 1), batch["bboxes"]), 1)
        targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
        gt_labels, gt_bboxes = targets.split((1, 4), 2)
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)

        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)

        _, target_bboxes, target_scores, fg_mask, target_gt_idx = self.assigner(
            pred_scores.detach().sigmoid(),
            (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt,
        )

        target_scores_sum = max(target_scores.sum(), 1)

        # Cls loss — normalize by nc so TAL quality is nc-invariant.
        # BCE sums over B*HW*nc, so nc=1 gives ~3x weaker cls than nc=3.
        # Without this, weak cls → poor TAL → aux branch collapse for small nc.
        loss[1] = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum * (_CLS_NC_REF / self.nc)

        # Bbox loss
        if fg_mask.sum():
            box_loss, _ = self.bbox_loss(
                pred_distri, pred_bboxes, anchor_points,
                target_bboxes / stride_tensor, target_scores, target_scores_sum,
                fg_mask, imgsz, stride_tensor,
            )
            if self.use_bbox_loss:
                loss[0] = box_loss

        loss[0] *= self.hyp.box
        loss[1] *= self.hyp.cls
        # --- End inline TAL ---

        # Aux losses (simple mean reduction — stable gradients from all fg anchors)
        aux_losses = self._compute_aux_losses(aux_preds, batch, target_gt_idx, fg_mask)
        for i, k in enumerate(["lr_distance", "depth", "dimensions", "orientation"], 2):
            if k in aux_losses:
                loss[i] = aux_losses[k] * float(self.aux_w.get(k, 1.0))

        return loss * batch_size, loss.detach()


class Stereo3DDetE2ELoss(E2ELoss):
    """E2E loss wrapper for stereo 3D detection.

    Logs one2many losses (which include aux branches) instead of one2one.
    Handles flat preds from eval mode (validation) by falling back to one2many loss only.
    """

    def __call__(self, preds, batch):
        """Compute weighted E2E loss, logging one2many losses (has all 6 components)."""
        preds = self.one2many.parse_output(preds)
        # Eval forward returns flat preds (no one2many/one2one wrapper)
        if "one2many" not in preds:
            return self.one2many.loss(preds, batch)
        loss_o2m = self.one2many.loss(preds["one2many"], batch)
        loss_o2o = self.one2one.loss(preds["one2one"], batch)
        return loss_o2m[0] * self.o2m + loss_o2o[0] * self.o2o, loss_o2m[1]
