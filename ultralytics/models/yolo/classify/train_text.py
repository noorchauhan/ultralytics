# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from ultralytics.models.yolo.classify.train import ClassificationTrainer
from ultralytics.nn.tasks import TextClassificationModel
from ultralytics.utils import DEFAULT_CFG, LOGGER, RANK, TQDM


class TextClassificationTrainer(ClassificationTrainer):
    """Trainer for text-aligned classification pre-training with MobileCLIP2 (https://arxiv.org/abs/2508.20691).

    Extend ClassificationTrainer with three loss modes for text-supervised training: 'contrastive' (CE + CLIP-style
    cosine similarity), 'text_similarity' (CE + KL from text embedding structure), and 'clip_distill' (CE + KL from
    pre-computed MobileCLIP2 image embeddings following dataset reinforcement https://arxiv.org/abs/2407.10886).

    Attributes:
        text_embeddings (torch.Tensor): Pre-computed (nc, embed_dim) text embeddings.
        text_similarity (torch.Tensor): Pre-computed (nc, nc) text similarity matrix.
        teacher_img_embeds (torch.Tensor): Pre-computed (N, embed_dim) MobileCLIP2 image embeddings.
        loss_mode (str): Active loss mode.
        teacher_variant (str): MobileCLIP2 variant for teacher ('s4' or 'l14').
    """

    def __init__(self, cfg=DEFAULT_CFG, overrides: dict[str, Any] | None = None, _callbacks: dict | None = None):
        """Initialize TextClassificationTrainer.

        Args:
            cfg (dict[str, Any], optional): Default configuration dictionary.
            overrides (dict[str, Any], optional): Parameter overrides. Supports 'loss_mode' ('contrastive',
                'text_similarity', 'clip_distill') and 'teacher_variant' ('s4', 'l14').
            _callbacks (dict, optional): Callback functions.
        """
        if overrides is None:
            overrides = {}
        self.loss_mode = overrides.pop("loss_mode", "contrastive")
        self.teacher_variant = overrides.pop("teacher_variant", "s4")
        self.text_embeddings = None
        self.text_similarity = None
        self.teacher_img_embeds = None
        super().__init__(cfg, overrides, _callbacks)

    def get_model(self, cfg=None, weights=None, verbose: bool = True):
        """Return TextClassificationModel configured for text-aligned training.

        Args:
            cfg (Any, optional): Model configuration.
            weights (Any, optional): Pre-trained model weights.
            verbose (bool, optional): Whether to display model information.

        Returns:
            (TextClassificationModel): Model with projection head for text alignment.
        """
        model = TextClassificationModel(
            cfg,
            nc=self.data["nc"],
            ch=self.data["channels"],
            verbose=verbose and RANK == -1,
            loss_mode=self.loss_mode,
        )
        if weights:
            model.load(weights)
        for m in model.modules():
            if not self.args.pretrained and hasattr(m, "reset_parameters"):
                m.reset_parameters()
            if isinstance(m, torch.nn.Dropout) and self.args.dropout:
                m.p = self.args.dropout
        for p in model.parameters():
            p.requires_grad = True
        return model

    def build_dataset(self, img_path: str, mode: str = "train", batch=None):
        """Build dataset and set up text embeddings for training mode.

        Args:
            img_path (str): Path to dataset images.
            mode (str, optional): Dataset mode ('train', 'val', or 'test').
            batch (Any, optional): Batch information (unused).

        Returns:
            (ClassificationDataset): Dataset for the specified mode.
        """
        dataset = super().build_dataset(img_path, mode, batch)
        if mode == "train" and self.text_embeddings is None:
            self._setup_text_embeddings(Path(img_path).parent, dataset)
        return dataset

    def _setup_text_embeddings(self, cache_dir, dataset):
        """Pre-compute and cache text embeddings for all class names using MobileCLIP2.

        Args:
            cache_dir (Path): Directory to cache text embeddings.
            dataset (ClassificationDataset): Training dataset (passed for teacher pre-compute).
        """
        from ultralytics.nn.text_model import build_text_model

        variant = self.teacher_variant.lower().replace("-", "")
        cache_path = cache_dir / f"text_embeddings_mobileclip2_{variant}.pt"
        names = list(self.data["names"].values())

        if cache_path.exists():
            cached = torch.load(cache_path, map_location=self.device)
            if cached.get("names") == names:
                self.text_embeddings = cached["embeds"].to(self.device)
                LOGGER.info(f"Loaded cached text embeddings from {cache_path}")

        if self.text_embeddings is None:
            LOGGER.info(f"Generating text embeddings for {len(names)} classes with MobileCLIP2-{self.teacher_variant}")
            text_model = build_text_model(f"mobileclip2:{self.teacher_variant}", device=self.device)
            texts = [f"a photo of a {name}" for name in names]
            self.text_embeddings = text_model.encode_text(text_model.tokenize(texts)).detach()
            torch.save({"names": names, "embeds": self.text_embeddings.cpu()}, cache_path)
            del text_model

        self.text_similarity = self.text_embeddings @ self.text_embeddings.T
        self.model.text_similarity = self.text_similarity.to(self.device)

        if self.loss_mode == "clip_distill":
            self._load_teacher_embeddings(dataset)

    def _load_teacher_embeddings(self, dataset):
        """Load or generate pre-computed MobileCLIP2 image embeddings for all training images.

        Args:
            dataset (ClassificationDataset): Training dataset for pre-computing embeddings.
        """
        variant = self.teacher_variant.lower().replace("-", "")
        cache_path = Path(self.args.data) / f"teacher_img_embeds_mobileclip2_{variant}.pt"
        if not cache_path.exists():
            if RANK in {-1, 0}:
                self._precompute_teacher_embeddings(cache_path, dataset)
            if RANK >= 0:
                torch.distributed.barrier()
        self.teacher_img_embeds = torch.load(cache_path, map_location="cpu")
        LOGGER.info(f"Loaded teacher image embeddings: {self.teacher_img_embeds.shape} from {cache_path}")

    def _precompute_teacher_embeddings(self, cache_path, dataset):
        """Run MobileCLIP2 image encoder on all training images and save embeddings to disk.

        Image preprocessing uses CLIP-standard ImageNet normalization (not Ultralytics identity normalization).

        Args:
            cache_path (Path): Path to save the embeddings tensor.
            dataset (ClassificationDataset): Training dataset to read images from.
        """
        from PIL import Image

        from ultralytics.nn.image_model import build_image_model

        LOGGER.info(
            f"Pre-computing MobileCLIP2-{self.teacher_variant} image embeddings (one-time, ~30 min for ImageNet)..."
        )
        teacher = build_image_model(f"mobileclip2:{self.teacher_variant}", device=self.device)
        n = len(dataset)
        # Detect embed dim from a probe forward pass
        probe = teacher.encode_image(torch.randn(1, 3, 256, 256).to(self.device))
        embed_dim = probe.shape[-1]
        embeds = torch.zeros(n, embed_dim)
        batch_size = 64
        for i in TQDM(range(0, n, batch_size), desc="Teacher embeddings"):
            end = min(i + batch_size, n)
            batch_pil = [Image.open(dataset.samples[j][0]).convert("RGB") for j in range(i, end)]
            batch_tensors = torch.stack([teacher.image_preprocess(img) for img in batch_pil])
            batch_embeds = teacher.encode_image(batch_tensors.to(self.device))
            embeds[i:end] = batch_embeds.cpu()
        torch.save(embeds, cache_path)
        LOGGER.info(
            f"Saved teacher embeddings ({embeds.shape}, {cache_path.stat().st_size / 1e9:.2f}GB) to {cache_path}"
        )
        del teacher
        torch.cuda.empty_cache()

    def preprocess_batch(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Attach text embeddings and optional teacher embeddings to batch.

        Args:
            batch (dict[str, torch.Tensor]): Batch with 'img', 'cls', and 'idx' keys.

        Returns:
            (dict[str, torch.Tensor]): Batch with added 'txt_feats' and optionally 'teacher_img_embeds'.
        """
        batch = super().preprocess_batch(batch)
        batch["txt_feats"] = self.text_embeddings.to(device=batch["img"].device, dtype=batch["img"].dtype)
        if self.teacher_img_embeds is not None and "idx" in batch:
            batch["teacher_img_embeds"] = self.teacher_img_embeds[batch["idx"]].to(
                self.device, non_blocking=self.device.type == "cuda"
            )
        return batch

    def get_validator(self):
        """Return a validator that injects text embeddings into validation batches."""
        from copy import copy

        from ultralytics.models.yolo import classify

        self.loss_names = ["loss"]
        text_embeddings = self.text_embeddings

        class TextClassificationValidator(classify.ClassificationValidator):
            """Validator that attaches text embeddings to batches for text-aligned loss computation."""

            def preprocess(self, batch):
                """Preprocess batch and attach text embeddings matching image dtype."""
                batch = super().preprocess(batch)
                batch["txt_feats"] = text_embeddings.to(device=batch["img"].device, dtype=batch["img"].dtype)
                return batch

        return TextClassificationValidator(
            self.test_loader, self.save_dir, args=copy(self.args), _callbacks=self.callbacks
        )
