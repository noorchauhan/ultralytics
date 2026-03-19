# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from ultralytics.data.build import build_dataloader
from ultralytics.data.dataset import SemanticDataset
from ultralytics.engine.validator import BaseValidator
from ultralytics.utils import LOGGER, ops
from ultralytics.utils.metrics import SemanticMetrics


class SemanticValidator(BaseValidator):
    """Validator for semantic segmentation models.

    This validator evaluates semantic segmentation models using mIoU and pixel accuracy metrics.

    Attributes:
        metrics (SemanticMetrics): Metrics calculator for semantic segmentation.

    Examples:
        >>> from ultralytics.models.yolo.semantic import SemanticValidator
        >>> args = dict(model="yolo26n-semseg.pt", data="cityscapes8.yaml")
        >>> validator = SemanticValidator(args=args)
        >>> validator()
    """

    def __init__(self, dataloader=None, save_dir=None, args=None, _callbacks=None):
        """Initialize SemanticValidator.

        Args:
            dataloader (DataLoader, optional): DataLoader for validation.
            save_dir (Path, optional): Directory to save results.
            args (dict, optional): Arguments for the validator.
            _callbacks (list, optional): Callback functions.
        """
        super().__init__(dataloader, save_dir, args, _callbacks)
        self.args.task = "semantic"
        self.metrics = SemanticMetrics()

    def init_metrics(self, model):
        """Initialize metrics with model class names.

        Args:
            model (nn.Module): Model to validate.
        """
        self.names = model.names
        self.nc = len(self.names)
        self.metrics = SemanticMetrics(names=self.names)

    def preprocess(self, batch):
        """Preprocess a batch of images and masks.

        Args:
            batch (dict): Batch data containing images and masks.

        Returns:
            (dict): Preprocessed batch.
        """
        batch["img"] = batch["img"].to(self.device, non_blocking=True)
        batch["img"] = (batch["img"].half() if self.args.half else batch["img"].float()) / 255
        batch["semantic_mask"] = batch["semantic_mask"].to(self.device, non_blocking=True).long()
        return batch

    def postprocess(self, preds):
        """Convert logits to class predictions.

        Args:
            preds (torch.Tensor): Raw model output logits [B, nc, H, W].

        Returns:
            (torch.Tensor): Predicted class IDs [B, H, W].
        """
        if isinstance(preds, (tuple, list)):
            preds = preds[0]
        return preds.argmax(dim=1)

    def update_metrics(self, preds, batch):
        """Update metrics with predictions and ground truth.

        Args:
            preds (torch.Tensor): Predicted class IDs [B, H, W].
            batch (dict): Batch containing 'semantic_mask'.
        """
        targets = batch["semantic_mask"]
        # Resize preds to match target if needed (preds may be at stride-4 during training)
        if preds.shape[1:] != targets.shape[1:]:
            preds = F.interpolate(preds.float().unsqueeze(1), targets.shape[1:], mode="nearest").squeeze(1).long()
        self.metrics.process(preds.cpu().numpy(), targets.cpu().numpy())

    def get_stats(self):
        """Return validation statistics.

        Returns:
            (dict): Dictionary of validation metrics.
        """
        return self.metrics.results_dict

    def get_desc(self):
        """Return a formatted description of evaluation metrics.

        Returns:
            (str): Formatted string with metric names.
        """
        return ("%22s" + "%11s" * 2) % ("Class", "mIoU", "PixAcc")

    def print_results(self):
        """Print validation results including per-class IoU."""
        pf = "%22s" + "%11.4f" * 2  # print format
        LOGGER.info(pf % ("all", self.metrics.miou, self.metrics.pixel_accuracy))
        # Per-class IoU
        per_class = self.metrics.per_class_iou
        for i, name in self.names.items():
            if i < len(per_class):
                LOGGER.info(f"  {name}: IoU={per_class[i]:.4f}")

    def build_dataset(self, img_path, mode="val", batch=None):
        """Build semantic segmentation dataset.

        Args:
            img_path (str): Path to images.
            mode (str): Dataset mode.
            batch (int, optional): Batch size.

        Returns:
            (SemanticDataset): Dataset object.
        """
        use_rect = mode == "val" and self.args.rect
        return SemanticDataset(
            img_path=img_path,
            imgsz=self.args.imgsz,
            augment=False,
            hyp=self.args,
            data=self.data,
            rect=use_rect,
            batch_size=batch,
            stride=self.stride,
            prefix=f"{mode}: ",
        )

    def get_dataloader(self, dataset_path, batch_size=16):
        """Build and return a dataloader for semantic segmentation validation.

        Args:
            dataset_path (str): Path to the dataset.
            batch_size (int): Batch size.

        Returns:
            (DataLoader): Validation dataloader.
        """
        dataset = self.build_dataset(dataset_path, batch=batch_size)
        return build_dataloader(dataset, batch=batch_size, workers=self.args.workers * 2, shuffle=False, rank=-1)
