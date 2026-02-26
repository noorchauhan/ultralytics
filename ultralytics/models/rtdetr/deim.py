# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

import math
import random
from copy import copy
from typing import Any

import torch
from torch import optim

from ultralytics.data import YOLODataset
from ultralytics.data.augment import Compose, Format
from ultralytics.utils import LOGGER, colorstr
from ultralytics.utils.torch_utils import one_cycle

from .detr_augment import compute_policy_epochs, rtdetr_deim_transforms
from .train import RTDETRTrainer
from .val import RTDETRDataset, RTDETRValidator

__all__ = ("RTDETRDEIMDataset", "RTDETRDEIMValidator", "RTDETRDEIMTrainer")


class _RTDETRDEIMBatchAugment:
    """Batch-level DEIM augmentations (MixUp + CopyBlend) applied in collate_fn."""

    _COPYBLEND_AREA_THRESHOLD = 100.0
    _COPYBLEND_NUM_OBJECTS = 3
    _COPYBLEND_RANDOM_NUM_OBJECTS = False
    _COPYBLEND_TYPE = "blend"
    _COPYBLEND_WITH_EXPAND = True
    _COPYBLEND_EXPAND_RATIOS = (0.1, 0.25)

    def __init__(
        self,
        mixup_prob: float,
        mixup_epochs: tuple[int, int],
        copyblend_prob: float,
        copyblend_epochs: tuple[int, int],
    ) -> None:
        self.mixup_prob = mixup_prob
        self.mixup_epochs = mixup_epochs
        self.copyblend_prob = copyblend_prob
        self.copyblend_epochs = copyblend_epochs
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        """Update current epoch for DEIM batch augmentation scheduling."""
        self.epoch = epoch

    def __call__(self, batch: list[dict]) -> dict:
        new_batch = YOLODataset.collate_fn(batch)
        mixup_start, mixup_stop = self.mixup_epochs
        copyblend_start, copyblend_stop = self.copyblend_epochs
        if mixup_start <= self.epoch < mixup_stop and random.random() < self.mixup_prob:
            new_batch = self._apply_mixup(new_batch)
        # DEIMv2 applies either MixUp or CopyBlend for a batch, not both.
        elif copyblend_start <= self.epoch < copyblend_stop and random.random() < self.copyblend_prob:
            new_batch = self._apply_copyblend(new_batch)
        return new_batch

    @staticmethod
    def _boxes_area_xywhn(boxes: torch.Tensor, w: int, h: int) -> torch.Tensor:
        """Compute absolute area from normalized xywh boxes."""
        if boxes.numel() == 0:
            return boxes.new_zeros((0,), dtype=torch.float32)
        return boxes[:, 2].to(torch.float32) * float(w) * boxes[:, 3].to(torch.float32) * float(h)

    @staticmethod
    def _stack_or_empty(tensors: list[torch.Tensor], shape: tuple[int, ...], *, like: torch.Tensor) -> torch.Tensor:
        """Stack or return empty tensor with desired shape/dtype/device."""
        if tensors:
            return torch.cat(tensors, dim=0)
        return torch.empty(shape, device=like.device, dtype=like.dtype)

    def _batch_to_targets(self, batch: dict) -> list[dict[str, torch.Tensor]]:
        """Convert flattened Ultralytics target format into DEIMv2-style per-image targets."""
        images = batch["img"]
        bs, _, h, w = images.shape
        bboxes = batch["bboxes"]
        cls = batch["cls"]
        batch_idx = batch["batch_idx"].view(-1).to(dtype=torch.long)
        labels_flat = cls.view(-1)
        mixup_flat = batch.get("mixup")
        mixup_flat = mixup_flat.view(-1) if isinstance(mixup_flat, torch.Tensor) else None

        targets = []
        for i in range(bs):
            mask = batch_idx == i
            boxes_i = bboxes[mask]
            labels_i = labels_flat[mask]
            area_i = self._boxes_area_xywhn(boxes_i, w=w, h=h)
            target = {"boxes": boxes_i.clone(), "labels": labels_i.clone(), "area": area_i}
            if mixup_flat is not None and mixup_flat.numel() == bboxes.shape[0]:
                target["mixup"] = mixup_flat[mask].clone()
            targets.append(target)
        return targets

    def _targets_to_batch(self, batch: dict, targets: list[dict[str, torch.Tensor]]) -> dict:
        """Convert DEIMv2-style per-image targets back to flattened Ultralytics format."""
        ref_boxes = batch["bboxes"]
        ref_cls = batch["cls"]
        ref_batch_idx = batch["batch_idx"]

        boxes_list, cls_list, batch_idx_list, mixup_list = [], [], [], []
        has_mixup = any("mixup" in t for t in targets)

        for i, target in enumerate(targets):
            boxes = target["boxes"]
            n = int(boxes.shape[0])
            if n == 0:
                continue

            labels = target["labels"]
            labels = labels.view(-1, 1) if ref_cls.ndim == 2 else labels.view(-1)

            boxes_list.append(boxes.to(device=ref_boxes.device, dtype=ref_boxes.dtype))
            cls_list.append(labels.to(device=ref_cls.device, dtype=ref_cls.dtype))
            batch_idx_list.append(torch.full((n,), i, device=ref_batch_idx.device, dtype=ref_batch_idx.dtype))

            if has_mixup:
                if "mixup" in target:
                    mixup_vals = target["mixup"]
                else:
                    mixup_vals = torch.ones((n,), device=ref_boxes.device, dtype=torch.float32)
                mixup_list.append(mixup_vals.to(device=ref_boxes.device, dtype=torch.float32))

        batch["bboxes"] = self._stack_or_empty(boxes_list, (0, ref_boxes.shape[1]), like=ref_boxes)
        if ref_cls.ndim == 2:
            batch["cls"] = self._stack_or_empty(cls_list, (0, ref_cls.shape[1]), like=ref_cls)
        else:
            batch["cls"] = self._stack_or_empty(cls_list, (0,), like=ref_cls)
        batch["batch_idx"] = self._stack_or_empty(batch_idx_list, (0,), like=ref_batch_idx)

        if has_mixup:
            batch["mixup"] = self._stack_or_empty(mixup_list, (0,), like=ref_boxes).to(torch.float32)
        elif "mixup" in batch:
            batch.pop("mixup")

        return batch

    def _apply_mixup(self, batch: dict) -> dict:
        images = batch["img"]
        bs = images.shape[0]
        if bs < 2:
            return batch

        targets = self._batch_to_targets(batch)
        shifted_targets = targets[-1:] + targets[:-1]
        updated_targets = []

        beta = round(random.uniform(0.45, 0.55), 6)
        images_f = images.to(torch.float32)
        shifted_images = torch.roll(images_f, shifts=1, dims=0)
        batch["img"] = shifted_images.mul(1.0 - beta).add(images_f.mul(beta))

        for target, shifted_target in zip(targets, shifted_targets):
            out = {
                "boxes": torch.cat([target["boxes"], shifted_target["boxes"]], dim=0),
                "labels": torch.cat([target["labels"], shifted_target["labels"]], dim=0),
                "area": torch.cat([target["area"], shifted_target["area"]], dim=0),
                "mixup": torch.tensor(
                    [beta] * len(target["labels"]) + [1.0 - beta] * len(shifted_target["labels"]),
                    device=batch["img"].device,
                    dtype=torch.float32,
                ),
            }
            updated_targets.append(out)

        return self._targets_to_batch(batch, updated_targets)

    def _apply_copyblend(self, batch: dict) -> dict:
        """CopyBlend implementation aligned with DEIMv2 collate behavior."""
        images = batch["img"]
        bs = images.shape[0]
        if bs < 2:
            return batch

        images_f = images.to(torch.float32)
        targets = self._batch_to_targets(batch)
        beta = round(random.uniform(0.45, 0.55), 6)
        img_height, img_width = images_f[0].shape[-2:]

        objects_pool: dict[str, list[Any]] = {
            "boxes": [],
            "labels": [],
            "areas": [],
            "image_idx": [],
            "image_height": [],
            "image_width": [],
        }

        for i in range(bs):
            source_boxes = targets[i]["boxes"]
            source_labels = targets[i]["labels"]
            source_areas = targets[i]["area"]

            valid_objects = [idx for idx in range(len(source_boxes)) if source_areas[idx] >= self._COPYBLEND_AREA_THRESHOLD]
            for idx in valid_objects:
                objects_pool["boxes"].append(source_boxes[idx])
                objects_pool["labels"].append(source_labels[idx])
                objects_pool["areas"].append(source_areas[idx])
                objects_pool["image_idx"].append(i)
                objects_pool["image_height"].append(img_height)
                objects_pool["image_width"].append(img_width)

        if len(objects_pool["boxes"]) == 0:
            return batch

        for key in ["boxes", "labels", "areas"]:
            objects_pool[key] = torch.stack(objects_pool[key]) if objects_pool[key] else torch.tensor([])

        updated_images = images_f.clone()
        updated_targets = [
            {k: (v.clone() if isinstance(v, torch.Tensor) else v) for k, v in target.items()} for target in targets
        ]

        for i in range(bs):
            pool_size = len(objects_pool["boxes"])
            if self._COPYBLEND_RANDOM_NUM_OBJECTS:
                num_objects = random.randint(1, min(self._COPYBLEND_NUM_OBJECTS, pool_size))
            else:
                num_objects = min(self._COPYBLEND_NUM_OBJECTS, pool_size)

            selected_indices = random.sample(range(pool_size), num_objects)
            blend_boxes, blend_labels, blend_areas, blend_mixup_ratios = [], [], [], []

            for idx in selected_indices:
                box = objects_pool["boxes"][idx]
                label = objects_pool["labels"][idx]
                area = objects_pool["areas"][idx]
                source_idx = objects_pool["image_idx"][idx]
                source_height = objects_pool["image_height"][idx]
                source_width = objects_pool["image_width"][idx]

                cx, cy, bw, bh = box.tolist()
                x1_src = int((cx - bw / 2) * source_width)
                y1_src = int((cy - bh / 2) * source_height)
                x2_src = int((cx + bw / 2) * source_width)
                y2_src = int((cy + bh / 2) * source_height)

                x1_src = max(x1_src, 0)
                y1_src = max(y1_src, 0)
                x2_src = min(x2_src, img_width)
                y2_src = min(y2_src, img_height)
                new_w_px = x2_src - x1_src
                new_h_px = y2_src - y1_src
                if new_w_px <= 0 or new_h_px <= 0:
                    continue

                x1 = random.randint(0, img_width - new_w_px) if new_w_px < img_width else 0
                y1 = random.randint(0, img_height - new_h_px) if new_h_px < img_height else 0
                x2, y2 = x1 + new_w_px, y1 + new_h_px

                new_cx = (x1 + new_w_px / 2) / img_width
                new_cy = (y1 + new_h_px / 2) / img_height
                new_w = new_w_px / img_width
                new_h = new_h_px / img_height

                blend_boxes.append(torch.tensor([new_cx, new_cy, new_w, new_h], device=box.device, dtype=box.dtype))
                blend_labels.append(label)
                blend_areas.append(area)
                blend_mixup_ratios.append(1.0 - beta)

                if self._COPYBLEND_WITH_EXPAND:
                    alpha = round(random.uniform(self._COPYBLEND_EXPAND_RATIOS[0], self._COPYBLEND_EXPAND_RATIOS[1]), 6)
                    expand_w = int(new_w_px * alpha)
                    expand_h = int(new_h_px * alpha)

                    x1_expand = x1_src - max(x1_src - expand_w, 0)
                    y1_expand = y1_src - max(y1_src - expand_h, 0)
                    x2_expand = min(x2_src + expand_w, img_width) - x2_src
                    y2_expand = min(y2_src + expand_h, img_height) - y2_src

                    new_x1_expand = x1 - max(x1 - x1_expand, 0)
                    new_y1_expand = y1 - max(y1 - y1_expand, 0)
                    new_x2_expand = min(x2 + x2_expand, img_width) - x2
                    new_y2_expand = min(y2 + y2_expand, img_height) - y2

                    x1_src, y1_src = x1_src - new_x1_expand, y1_src - new_y1_expand
                    x2_src, y2_src = x2_src + new_x2_expand, y2_src + new_y2_expand
                    x1, y1 = x1 - new_x1_expand, y1 - new_y1_expand
                    x2, y2 = x2 + new_x2_expand, y2 + new_y2_expand

                copy_patch_orig = images_f[source_idx, :, y1_src:y2_src, x1_src:x2_src]
                if self._COPYBLEND_TYPE == "blend":
                    blended_patch = updated_images[i, :, y1:y2, x1:x2] * beta + copy_patch_orig * (1 - beta)
                    updated_images[i, :, y1:y2, x1:x2] = blended_patch
                else:
                    updated_images[i, :, y1:y2, x1:x2] = copy_patch_orig

            if blend_boxes:
                blend_boxes_t = torch.stack(blend_boxes)
                blend_labels_t = torch.stack(blend_labels)
                blend_areas_t = torch.stack(blend_areas)

                updated_targets[i]["mixup"] = torch.tensor(
                    [1.0] * len(updated_targets[i]["boxes"]) + blend_mixup_ratios,
                    device=blend_boxes_t.device,
                    dtype=torch.float32,
                )
                updated_targets[i]["boxes"] = torch.cat([updated_targets[i]["boxes"], blend_boxes_t])
                updated_targets[i]["labels"] = torch.cat([updated_targets[i]["labels"], blend_labels_t])
                updated_targets[i]["area"] = torch.cat([updated_targets[i]["area"], blend_areas_t])

        batch["img"] = updated_images
        return self._targets_to_batch(batch, updated_targets)


class RTDETRDEIMDataset(RTDETRDataset):
    """RT-DETR dataset variant that uses a dedicated DEIM augmentation pipeline."""

    def __init__(self, *args, data=None, **kwargs):
        hyp = kwargs["hyp"]
        self.policy_epochs, self.mixup_epochs, self.copyblend_epochs = self._compute_deim_schedule(hyp)
        self.mosaic_prob = float(hyp.mosaic)
        self.mixup_prob = float(hyp.mixup)
        self.copyblend_prob = float(hyp.copy_paste)
        super().__init__(*args, data=data, **kwargs)
        if self.augment:
            if self.mixup_prob > 0.0 or self.copyblend_prob > 0.0:
                self.collate_fn = _RTDETRDEIMBatchAugment(
                    mixup_prob=self.mixup_prob,
                    mixup_epochs=self.mixup_epochs,
                    copyblend_prob=self.copyblend_prob,
                    copyblend_epochs=self.copyblend_epochs,
                )
            self.set_epoch(0)

    def _compute_deim_schedule(self, hyp) -> tuple[tuple[int, int, int], tuple[int, int], tuple[int, int]]:
        """Compute DEIM stage boundaries from epochs only."""
        policy_epochs = compute_policy_epochs(hyp)
        mixup_epochs = policy_epochs[:2]
        copyblend_epochs = (policy_epochs[0], policy_epochs[2])
        return policy_epochs, mixup_epochs, copyblend_epochs

    def build_transforms(self, hyp=None):
        """Build DEIM transforms for train and standard formatting for train/val."""
        if self.augment:
            hyp.mixup = hyp.mixup if self.augment and not self.rect else 0.0
            hyp.cutmix = hyp.cutmix if self.augment and not self.rect else 0.0
            transforms = rtdetr_deim_transforms(
                self,
                self.imgsz,
                hyp,
                stretch=True,
                policy_epochs=self.policy_epochs,
                mosaic_prob=self.mosaic_prob,
            )
        else:
            transforms = Compose([])

        transforms.append(
            Format(
                bbox_format="xywh",
                normalize=True,
                return_mask=self.use_segments,
                return_keypoint=self.use_keypoints,
                batch_idx=True,
                mask_ratio=hyp.mask_ratio,
                mask_overlap=hyp.overlap_mask,
            )
        )
        return transforms

    def set_epoch(self, epoch: int) -> None:
        """Propagate epoch to transforms and collate_fn for DEIM stage scheduling."""
        self.epoch = epoch
        self.transforms.set_epoch(epoch)

        requires_collate_epoch = self.mixup_prob > 0.0 or self.copyblend_prob > 0.0
        if requires_collate_epoch:
            self.collate_fn.set_epoch(epoch)


class RTDETRDEIMValidator(RTDETRValidator):
    """Validator that builds the DEIM dataset variant."""

    def build_dataset(self, img_path, mode="val", batch=None):
        return RTDETRDEIMDataset(
            img_path=img_path,
            imgsz=self.args.imgsz,
            batch_size=batch,
            augment=False,
            hyp=self.args,
            rect=False,
            cache=self.args.cache or None,
            prefix=colorstr(f"{mode}: "),
            data=self.data,
        )


class RTDETRDEIMTrainer(RTDETRTrainer):
    """RT-DETR trainer variant with isolated DEIM augmentation scheduling."""
    _deim_callback_registered = False

    def build_dataset(self, img_path: str, mode: str = "val", batch: int | None = None):
        dataset = RTDETRDEIMDataset(
            img_path=img_path,
            imgsz=self.args.imgsz,
            batch_size=batch,
            augment=mode == "train",
            hyp=self.args,
            rect=False,
            cache=self.args.cache or None,
            single_cls=self.args.single_cls or False,
            prefix=colorstr(f"{mode}: "),
            classes=self.args.classes,
            data=self.data,
            fraction=self.args.fraction if mode == "train" else 1.0,
        )
        return dataset

    def _setup_scheduler(self):
        """Initialize LR scheduler with optional DEIM flat-cosine schedule."""
        scheduler_arg = self.args.lr_scheduler
        if scheduler_arg is None:
            return super()._setup_scheduler()
        scheduler_name = str(scheduler_arg).lower()
        if not scheduler_name:
            return super()._setup_scheduler()

        if scheduler_name in {"linear"}:
            self.lf = lambda x: max(1 - x / self.epochs, 0) * (1.0 - self.args.lrf) + self.args.lrf
        elif scheduler_name in {"cosine", "cos", "cos_lr"}:
            self.lf = one_cycle(1, self.args.lrf, self.epochs)
        elif scheduler_name in {"flatcosine", "flat_cosine", "flatcos"}:
            # Flat phase keeps LR constant, then cosine anneals to lrf.
            warmup_epochs = max(float(self.args.warmup_epochs), 0.0)
            default_flat_epoch = min(self.epochs, max(int(math.ceil(warmup_epochs)), 4) + self.epochs // 2)
            flat_epoch = default_flat_epoch if self.args.flat_epoch is None else int(self.args.flat_epoch)
            if not (0 <= flat_epoch <= self.epochs):
                raise ValueError(
                    f"flatcosine got invalid flat_epoch={flat_epoch} for epochs={self.epochs}. "
                    "Expected 0 <= flat_epoch <= epochs."
                )
            gamma = float(self.args.lrf)
            if not (0.0 <= gamma <= 1.0):
                raise ValueError(f"flatcosine got invalid lrf={gamma}. Expected 0.0 <= lrf <= 1.0.")
            decay_epochs = max(self.epochs - flat_epoch, 1)

            def _flat_cosine(epoch: int) -> float:
                if epoch < flat_epoch:
                    return 1.0
                progress = min(max((epoch - flat_epoch) / decay_epochs, 0.0), 1.0)
                return gamma + 0.5 * (1.0 - gamma) * (1.0 + math.cos(math.pi * progress))

            self.lf = _flat_cosine
        else:
            LOGGER.warning(f"Unknown lr_scheduler='{scheduler_name}', falling back to default scheduler.")
            return super()._setup_scheduler()

        self.scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=self.lf)

    def _on_train_epoch_start(self, trainer=None):
        """Apply DEIM epoch scheduling to transforms/collate and stop multi-scale at stage-4 start."""
        trainer = trainer or self
        epoch = trainer.epoch
        dataset = trainer.train_loader.dataset
        dataset.set_epoch(epoch)
        # InfiniteDataLoader keeps workers/iterator alive; reset so worker-side
        # dataset transforms and collate_fn pick up the updated epoch.
        trainer.train_loader.reset()
        stop_epoch = int(dataset.policy_epochs[-1])
        if epoch == stop_epoch and trainer.args.multi_scale > 0:
            trainer.args.multi_scale = 0.0
            LOGGER.info(f"DEIM stage-4 at epoch {epoch}: disabling multi-scale")

    def train(self, *args, **kwargs):
        # DEIM trainer handles augmentation schedule explicitly.
        # Keep base trainer's close_mosaic hook disabled to avoid overriding DEIM policy.
        if self.args.close_mosaic:
            self.args.close_mosaic = 0
        if not self._deim_callback_registered:
            self.add_callback("on_train_epoch_start", self._on_train_epoch_start)
            self._deim_callback_registered = True
        return super().train(*args, **kwargs)

    def get_validator(self):
        loss_names = ["giou_loss", "cls_loss", "l1_loss"]
        loss_gain = self.model_yaml.get("loss", {}).get("loss_gain", {})
        if "fgl" in loss_gain:
            loss_names.append("fgl_loss")
        if "ddf" in loss_gain:
            loss_names.append("ddf_loss")
        model = getattr(self.model, "module", self.model)
        if getattr(model.model[-1], "one_to_many_groups", 0) > 0:
            loss_names.extend(["giou_o2m", "cls_o2m", "l1_o2m"])
        self.loss_names = tuple(loss_names)
        return RTDETRDEIMValidator(self.test_loader, save_dir=self.save_dir, args=copy(self.args))
