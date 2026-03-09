# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

import numpy as np
import torch

from ultralytics.engine.predictor import BasePredictor
from ultralytics.engine.results import Results
from ultralytics.utils import DEFAULT_CFG
from ultralytics.utils import ops


class SemanticPredictor(BasePredictor):
    """Predictor for semantic segmentation models.

    This predictor processes model outputs to produce per-pixel class label maps.

    Examples:
        >>> from ultralytics.models.yolo.semantic import SemanticPredictor
        >>> args = dict(model="yolo26n-semseg.pt", source="path/to/image.jpg")
        >>> predictor = SemanticPredictor(overrides=args)
        >>> predictor.predict_cli()
    """

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        """Initialize SemanticPredictor.

        Args:
            cfg (dict): Configuration for the predictor.
            overrides (dict, optional): Configuration overrides.
            _callbacks (list, optional): Callback functions.
        """
        super().__init__(cfg, overrides, _callbacks)
        self.args.task = "semantic"

    def postprocess(self, preds, img, orig_imgs):
        """Convert model logits to semantic segmentation results.

        Args:
            preds (torch.Tensor | tuple): Model output logits.
            img (torch.Tensor): Preprocessed input image tensor.
            orig_imgs (list | torch.Tensor): Original images.

        Returns:
            (list[Results]): List of Results objects with semantic masks.
        """
        if isinstance(preds, (tuple, list)):
            preds = preds[0]

        if not isinstance(orig_imgs, list):
            orig_imgs = [orig_imgs] if isinstance(orig_imgs, np.ndarray) else list(orig_imgs.cpu().numpy())

        results = []
        for i, (pred, orig_img) in enumerate(zip(preds, orig_imgs)):
            img_path = self.batch[0][i] if isinstance(self.batch[0], list) else self.batch[0]
            # pred: [nc, H, W] logits on letterboxed input. Remove padding, then resize to original image.
            oh, ow = orig_img.shape[:2]
            pred = ops.scale_masks(pred.unsqueeze(0), (oh, ow))[0]
            class_map = pred.argmax(0).cpu()  # [H, W]
            results.append(Results(orig_img, path=img_path, names=self.model.names, semantic_mask=class_map))
        return results
