"""Callback to add mixup/cutmix augmentation to classification preprocess_batch.

Monkey-patches trainer.preprocess_batch at train start so no source changes needed.
Works with both ClassificationTrainer and TextClassificationTrainer.

Usage:
    from callbacks import mixup
    model.add_callback("on_train_start", mixup.override(mixup=0.8, cutmix=1.0))
"""

import random
import types

import numpy as np
import torch


def override(mixup=0.0, cutmix=0.0):
    """Return on_train_start callback to patch preprocess_batch with mixup/cutmix.

    Args:
        mixup (float): Probability of applying mixup per batch.
        cutmix (float): Probability of applying cutmix per batch.
    """

    def callback(trainer):
        original = trainer.preprocess_batch

        def _preprocess_batch(self, batch):
            batch = original(batch)

            bs, _, h, w = batch["img"].shape
            use_cutmix = cutmix > 0 and (not mixup or random.random() > 0.5)
            if random.random() >= (cutmix if use_cutmix else mixup):
                return batch

            lam = np.random.beta(1.0, 1.0)
            idx = torch.arange(bs - 1, -1, -1, device=self.device)
            nc = self.data["nc"]

            cls_onehot = torch.zeros(bs, nc, device=self.device)
            cls_onehot.scatter_(1, batch["cls"].long().unsqueeze(1), 1.0)

            if use_cutmix:
                cut_ratio = np.sqrt(1.0 - lam)
                ch, cw = int(h * cut_ratio), int(w * cut_ratio)
                cy, cx = random.randint(0, h - 1), random.randint(0, w - 1)
                y1 = int(np.clip(cy - ch // 2, 0, h))
                y2 = int(np.clip(cy + ch // 2, 0, h))
                x1 = int(np.clip(cx - cw // 2, 0, w))
                x2 = int(np.clip(cx + cw // 2, 0, w))
                batch["img"][:, :, y1:y2, x1:x2] = batch["img"][idx, :, y1:y2, x1:x2]
                lam = 1.0 - (y2 - y1) * (x2 - x1) / (h * w)
            else:
                batch["img"] = lam * batch["img"] + (1.0 - lam) * batch["img"][idx]

            batch["cls"] = lam * cls_onehot + (1.0 - lam) * cls_onehot[idx]

            # Mix teacher embeddings for clip_distill compatibility
            if "teacher_img_embeds" in batch:
                batch["teacher_img_embeds"] = (
                    lam * batch["teacher_img_embeds"] + (1.0 - lam) * batch["teacher_img_embeds"][idx]
                )

            return batch

        trainer.preprocess_batch = types.MethodType(_preprocess_batch, trainer)

        original_plot = trainer.plot_training_samples

        def _plot_training_samples(self, batch, ni):
            if batch["cls"].ndim == 2:
                batch["cls"] = batch["cls"].argmax(dim=1)
            original_plot(batch, ni)

        trainer.plot_training_samples = types.MethodType(_plot_training_samples, trainer)

    return callback
