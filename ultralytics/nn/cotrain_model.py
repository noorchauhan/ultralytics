"""Co-Training model wrapper for mutual learning between teacher and student.

Both models are trained simultaneously on the same data, with bidirectional
feature distillation loss. Each model can reside on a different GPU device.
"""

import math
from copy import deepcopy

import torch
import torch.nn.functional as F
from torch import nn

from ultralytics.utils import LOGGER
from ultralytics.utils.metrics import bbox_iou
from ultralytics.utils.tal import dist2bbox
from ultralytics.utils.torch_utils import copy_attr

from .tasks import DetectionModel, load_checkpoint


class CoTrainingModel(nn.Module):
    """Co-Training model that trains both teacher and student simultaneously.

    Unlike DistillationModel which freezes the teacher, both models participate
    in training with bidirectional distillation loss. Each model can optionally
    be placed on a different GPU device.
    """

    def __init__(
        self,
        teacher_model: str | nn.Module,
        student_model: nn.Module,
        feats_idx: int | list[int],
        teacher_device: torch.device | str | int | None = None,
    ):
        """Initialize CoTrainingModel.

        Args:
            teacher_model: Path to teacher weights or an nn.Module.
            student_model: The student model (already on its device).
            feats_idx: Layer index (or list of indices) to extract features from.
            teacher_device: Device for the teacher model. If None, same device as student.
        """
        super().__init__()
        if isinstance(feats_idx, int):
            feats_idx = [feats_idx]

        self.student_device = next(student_model.parameters()).device
        self.teacher_device = self._normalize_device(teacher_device, fallback=self.student_device)
        teacher_model = self._build_teacher_model(teacher_model, student_model)

        self.teacher_model = teacher_model.to(self.teacher_device)
        self.student_model = student_model  # already on student_device
        self.feats_idx = feats_idx
        self.inference_target = "student"  # inference routing target: "student" or "teacher"

        # Align teacher runtime attrs to current training config from student.
        student_core = self._student()
        teacher_core = self._teacher()
        if hasattr(student_core, "args"):
            teacher_core.args = student_core.args
        if hasattr(student_core, "names"):
            teacher_core.names = student_core.names
        if hasattr(student_core, "nc"):
            teacher_core.nc = student_core.nc
        # Rebuild teacher loss with current training args on first teacher loss() call.
        if hasattr(teacher_core, "criterion"):
            teacher_core.criterion = None

        # Probe feature dimensions (no grad needed for init)
        with torch.inference_mode():
            dummy_student = torch.zeros(1, 3, 256, 256, device=self.student_device)
            dummy_teacher = torch.zeros(1, 3, 256, 256, device=self.teacher_device)
            teacher_output = teacher_model(dummy_teacher, embed=feats_idx, direct_return=True)
            student_output = student_model(dummy_student, embed=feats_idx, direct_return=True)
            assert len(teacher_output) == len(student_output), "Feature dimensions must match in length."

        # Copy student attributes (stride, names, nc, etc.) to self
        copy_attr(self, student_model)

        # Cache distillation config
        args = self._student().args
        self.distill_box_loss = args.distill_box_loss
        self.distill_cls_loss = args.distill_cls_loss
        self.distill_feature_loss = args.distill_feature_loss
        self.distill_box = args.distill_box
        self.distill_cls = args.distill_cls
        self.distill_feature = args.distill_feature
        self.cur_epoch = 0
        self.total_epochs = max(int(getattr(args, "epochs", 1) or 1), 1)

        # Build projectors (student→teacher dimension alignment)
        if self.distill_feature_loss:
            projectors = []
            for s_out, t_out in zip(student_output, teacher_output):
                s_dim = self._decouple(s_out, shape_check=True).shape[1]
                t_dim = self._decouple(t_out, shape_check=True).shape[1]
                if args.distill_projector == "linear":
                    projectors.append(nn.Conv2d(s_dim, t_dim, 1))
                elif args.distill_projector == "mlp_silu":
                    projectors.append(
                        nn.Sequential(
                            nn.Conv2d(s_dim, t_dim, kernel_size=1, stride=1, padding=0),
                            nn.SiLU(),
                            nn.Conv2d(t_dim, t_dim, kernel_size=1, stride=1, padding=0),
                        )
                    )
                else:
                    projectors.append(
                        nn.Sequential(
                            nn.Conv2d(s_dim, t_dim, 1),
                            nn.ReLU(inplace=True),
                            nn.Conv2d(t_dim, t_dim, 1),
                        )
                    )
            # Projector lives on student device (student feats are projected to teacher dim)
            self.projector = nn.ModuleList(projectors)

        if self.distill_feature_loss == "mgd":
            generations = []
            for t_out in teacher_output:
                if not isinstance(t_out, dict):
                    t_dim = t_out.shape[1]
                    generations.append(
                        nn.Sequential(
                            nn.Conv2d(t_dim, t_dim, 3, padding=1),
                            nn.SiLU(),
                            nn.Conv2d(t_dim, t_dim, 3, padding=1),
                        )
                    )
            self.generation = nn.ModuleList(generations)

        if self.distill_feature_loss == "cwd":
            norms = []
            for t_out in teacher_output:
                if not isinstance(t_out, dict):
                    t_dim = t_out.shape[1]
                    norms.append(nn.BatchNorm2d(t_dim, affine=False))
            self.norm = nn.ModuleList(norms)

        # Ensure head output layer is included for cls/box distillation
        if self.distill_box or self.distill_cls or self.distill_feature_loss == "sl2":
            if 23 not in self.feats_idx:
                self.feats_idx = list(self.feats_idx)
                self.feats_idx.append(23)
                self.feats_idx = tuple(self.feats_idx)

        self.distill_area = args.distill_area
        self.distill_branch = args.distill_branch.split(",")
        for b in self.distill_branch:
            assert b in {"one2one", "one2many"}, f"Unknown branch: {b}"
        self.refresh_devices(self.student_device, self.teacher_device)

    @classmethod
    def _build_teacher_model(cls, teacher_model: str | nn.Module, student_model: nn.Module) -> nn.Module:
        """Rebuild teacher with the current training nc/ch, then load compatible weights."""
        teacher_source = load_checkpoint(teacher_model)[0] if isinstance(teacher_model, str) else teacher_model
        teacher_source = cls._unwrap_if_ddp(teacher_source)
        student_core = cls._unwrap_if_ddp(student_model)

        if not isinstance(teacher_source, nn.Module):
            raise TypeError(f"Expected teacher_model to resolve to nn.Module, got {type(teacher_source).__name__}")
        if getattr(teacher_source, "task", None) not in {None, "detect"}:
            raise ValueError(
                f"CoTrainingModel only supports detection teachers for rebuild, got task={teacher_source.task!r}."
            )
        if not hasattr(teacher_source, "yaml") or teacher_source.yaml is None:
            raise ValueError("Teacher model must expose a valid model.yaml to rebuild the detection head.")

        student_nc = getattr(student_core, "nc", None)
        if student_nc is None:
            raise ValueError("Student model must define 'nc' before constructing CoTrainingModel.")

        teacher_cfg = deepcopy(teacher_source.yaml)
        student_ch = teacher_cfg.get("channels", 3)
        if hasattr(student_core, "yaml") and isinstance(student_core.yaml, dict):
            student_ch = student_core.yaml.get("channels", student_ch)

        rebuilt_teacher = DetectionModel(cfg=teacher_cfg, nc=student_nc, ch=student_ch, verbose=False)
        if hasattr(student_core, "args"):
            rebuilt_teacher.args = student_core.args
        rebuilt_teacher.load(teacher_source, verbose=False)

        source_nc = getattr(teacher_source, "nc", teacher_cfg.get("nc"))
        LOGGER.info(
            "Rebuilt co-training teacher with current training head "
            f"(source_nc={source_nc}, target_nc={student_nc}) and loaded shape-compatible weights"
        )
        return rebuilt_teacher

    @staticmethod
    def _normalize_device(device, fallback=None):
        """Normalize device arguments into torch.device."""
        if device is None:
            return fallback if fallback is not None else torch.device("cpu")
        if isinstance(device, torch.device):
            return device
        if isinstance(device, int):
            return torch.device(f"cuda:{device}")
        if isinstance(device, str) and device.isdigit():
            return torch.device(f"cuda:{device}")
        return torch.device(device)

    @staticmethod
    def _unwrap_if_ddp(module):
        """Get inner module if wrapped by DistributedDataParallel."""
        return module.module if isinstance(module, nn.parallel.DistributedDataParallel) else module

    def _student(self):
        """Get inner student model regardless of DDP wrapping."""
        return self._unwrap_if_ddp(self.student_model)

    def _teacher(self):
        """Get inner teacher model regardless of DDP wrapping."""
        return self._unwrap_if_ddp(self.teacher_model)

    @staticmethod
    def _get_tensor_device(obj):
        """Return first tensor device found in nested containers."""
        if isinstance(obj, torch.Tensor):
            return obj.device
        if isinstance(obj, dict):
            for v in obj.values():
                dev = CoTrainingModel._get_tensor_device(v)
                if dev is not None:
                    return dev
            return None
        if isinstance(obj, (list, tuple)):
            for v in obj:
                dev = CoTrainingModel._get_tensor_device(v)
                if dev is not None:
                    return dev
        return None

    @staticmethod
    def _to_device(obj, device):
        """Recursively move tensors in nested containers to target device."""
        if isinstance(obj, torch.Tensor):
            return obj.to(device)
        if isinstance(obj, dict):
            return {k: CoTrainingModel._to_device(v, device) for k, v in obj.items()}
        if isinstance(obj, list):
            return [CoTrainingModel._to_device(v, device) for v in obj]
        if isinstance(obj, tuple):
            return tuple(CoTrainingModel._to_device(v, device) for v in obj)
        return obj

    @staticmethod
    def _detach(obj):
        """Recursively detach tensors in nested containers from autograd graph."""
        if isinstance(obj, torch.Tensor):
            return obj.detach()
        if isinstance(obj, dict):
            return {k: CoTrainingModel._detach(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [CoTrainingModel._detach(v) for v in obj]
        if isinstance(obj, tuple):
            return tuple(CoTrainingModel._detach(v) for v in obj)
        return obj

    def refresh_devices(self, student_device=None, teacher_device=None):
        """Refresh device cache and move student/teacher and local distill modules to expected devices."""
        if student_device is None:
            student_device = next(self._unwrap_if_ddp(self.student_model).parameters()).device
        student_device = self._normalize_device(student_device)
        teacher_device = self._normalize_device(teacher_device, fallback=student_device)

        self.student_device = student_device
        self.teacher_device = teacher_device

        self._unwrap_if_ddp(self.student_model).to(student_device)
        self._unwrap_if_ddp(self.teacher_model).to(teacher_device)
        for name in ("projector", "generation", "norm"):
            module = getattr(self, name, None)
            if isinstance(module, nn.Module):
                module.to(student_device)
        return self

    def local_distill_parameters(self):
        """Return local distillation-module parameters that are not inside student/teacher DDP wrappers."""
        params = []
        for name in ("projector", "generation", "norm"):
            module = getattr(self, name, None)
            if isinstance(module, nn.Module):
                params.extend(p for p in module.parameters() if p.requires_grad)
        return params

    def _set_epoch_progress(self, cur_epoch: int, total_epochs: int):
        """Update cached epoch progress for distillation scheduling."""
        self.cur_epoch = int(cur_epoch)
        self.total_epochs = max(int(total_epochs), 1)

    def train(self, mode: bool = True):
        """Set both models to train mode."""
        super().train(mode)
        # Both teacher and student are trainable in co-training
        self.teacher_model.train(mode)
        self.student_model.train(mode)
        return self

    def set_inference_target(self, target: str = "student"):
        """Set inference routing target. Supported values: 'student', 'teacher'."""
        if target not in {"student", "teacher"}:
            raise ValueError(f"Unknown inference target: {target}. Expected 'student' or 'teacher'.")
        self.inference_target = target
        return self

    # ------------------------------------------------------------------
    # Forward / prediction
    # ------------------------------------------------------------------

    def forward(self, x, *args, **kwargs):
        """Forward pass – routes to loss() during training, predict() during inference."""
        if isinstance(x, dict):  # training / val-while-training
            return self.loss(x, *args, **kwargs)
        if self.inference_target == "teacher":
            src_device = self._get_tensor_device(x)
            x_teacher = self._to_device(x, self.teacher_device)
            preds = self._teacher().predict(x_teacher, *args, **kwargs)
            return self._to_device(preds, src_device) if src_device is not None else preds
        return self._student().predict(x, *args, **kwargs)

    # ------------------------------------------------------------------
    # Loss computation
    # ------------------------------------------------------------------

    def loss(self, batch, preds=None):
        """Compute co-training loss.

        Total loss = student_detection_loss + teacher_detection_loss + distill_loss (teacher targets detached)
        """
        student_model = self._student()
        teacher_model = self._teacher()
        student_dev = batch["img"].device
        teacher_dev = self.teacher_device

        if not self.training:
            # Validation: compute student + teacher detection losses for consistent loss vector length.
            img = batch["img"]
            preds_student = self.student_model(img)
            student_loss, student_loss_detach = student_model.loss(batch, preds_student)

            batch_teacher = {k: self._to_device(v, teacher_dev) for k, v in batch.items()}
            preds_teacher = self.teacher_model(batch_teacher["img"])
            teacher_loss, teacher_loss_detach = teacher_model.loss(batch_teacher, preds_teacher)
            teacher_loss = teacher_loss.to(student_dev)
            teacher_loss_detach = teacher_loss_detach.to(student_dev)

            zero = torch.zeros(1, device=student_dev)
            total_loss = torch.cat([student_loss, teacher_loss, zero])
            total_loss_detach = torch.cat([student_loss_detach, teacher_loss_detach, zero])
            return total_loss, total_loss_detach

        img = batch["img"]  # on student device

        # Build teacher batch (move all tensors to teacher device)
        batch_teacher = {}
        for k, v in batch.items():
            batch_teacher[k] = self._to_device(v, teacher_dev)

        # ---- Student forward ----
        student_preds, student_feats = self.student_model(img, return_feats=True)
        student_loss, student_loss_detach, student_targets = student_model.loss(
            batch, student_preds, return_targets=True
        )

        # ---- Teacher forward ----
        teacher_preds, teacher_feats_raw = self.teacher_model(batch_teacher["img"], return_feats=True)
        teacher_loss, teacher_loss_detach = teacher_model.loss(batch_teacher, teacher_preds)

        # Move teacher loss to student device for aggregation
        teacher_loss = teacher_loss.to(student_dev)
        teacher_loss_detach = teacher_loss_detach.to(student_dev)

        # ---- Distillation loss (teacher targets, student-updated) ----
        loss_distill_cls = torch.zeros(1, device=student_dev)
        loss_distill_box = torch.zeros(1, device=student_dev)
        loss_distill_feature = torch.zeros(1, device=student_dev)

        # Precompute teacher scores for sl2/scosine
        teacher_scores = None
        if self.distill_feature_loss in {"sl2", "scosine"}:
            teacher_scores = self._compute_teacher_scores(teacher_feats_raw, target_device=student_dev)

        feature_main_masking = self.distill_area == "main" and self.distill_feature_loss in {"l1", "l2"}
        feature_fg_masks = None
        if feature_main_masking:
            feature_fg_masks = torch.split(student_targets["one2one"][0], [6400, 1600, 400], dim=-1)
        feature_level_idx = 0
        feature_weight = self.distill_feature

        for i, feat_idx in enumerate(self.feats_idx):
            # Teacher distill targets are detached: distill gradients flow to student side only.
            raw_teacher = self._detach(teacher_feats_raw[feat_idx])
            teacher_feat_moved = self._to_device(raw_teacher, student_dev)

            teacher_feat = self._decouple(teacher_feat_moved)
            student_feat = self._decouple(student_feats[feat_idx])

            assert isinstance(teacher_feat, type(student_feat)), (
                f"Type mismatch: teacher={type(teacher_feat)}, student={type(student_feat)}"
            )

            if isinstance(teacher_feat, dict):
                # Head output – distill cls and box
                for branch in self.distill_branch:
                    t_feat = self._decouple(teacher_feat_moved, branch=branch)
                    s_feat = self._decouple(student_feats[feat_idx], branch=branch)
                    assert "boxes" in t_feat and "scores" in t_feat

                    if self.distill_cls_loss:
                        t_logits = t_feat["scores"].permute(0, 2, 1).contiguous()
                        s_logits = s_feat["scores"].permute(0, 2, 1).contiguous()
                        if self.distill_area == "main":
                            fg_mask = student_targets[branch][0]
                            t_logits = t_logits[fg_mask]
                            s_logits = s_logits[fg_mask]
                        else:
                            t_logits = t_logits.view(-1, t_logits.shape[-1])
                            s_logits = s_logits.view(-1, s_logits.shape[-1])
                        loss_distill_cls += self._cls_kd_loss(s_logits, t_logits) * self.distill_cls

                    if self.distill_box_loss:
                        t_boxes = t_feat["boxes"]
                        s_boxes = s_feat["boxes"]
                        if self.distill_box_loss in {"iou", "siou"}:
                            anchor_points = student_targets[branch][3]
                            t_boxes = t_boxes.permute(0, 2, 1).contiguous()
                            s_boxes = s_boxes.permute(0, 2, 1).contiguous()
                            t_score_w = None
                            if self.distill_box_loss == "siou":
                                tscore = self._decouple(teacher_feat_moved, branch="one2one")["scores"]
                                score_w = tscore.sigmoid().amax(dim=1, keepdim=True)
                                t_score_w = score_w.permute(0, 2, 1).contiguous()
                            if self.distill_area == "main":
                                fg_mask = student_targets[branch][0]
                                t_boxes = t_boxes[fg_mask]
                                s_boxes = s_boxes[fg_mask]
                                anchor_points = anchor_points.unsqueeze(0).expand(fg_mask.shape[0], -1, -1)[fg_mask]
                                if t_score_w is not None:
                                    t_score_w = t_score_w[fg_mask]
                            loss_distill_box += self._box_kd_loss(s_boxes, t_boxes, anchor_points, t_score_w) * self.distill_box
                        else:
                            if self.distill_area == "main":
                                t_boxes = t_boxes.permute(0, 2, 1).contiguous()
                                s_boxes = s_boxes.permute(0, 2, 1).contiguous()
                                fg_mask = student_targets[branch][0]
                                t_boxes = t_boxes[fg_mask]
                                s_boxes = s_boxes[fg_mask]
                            loss_distill_box += self._box_kd_loss(s_boxes, t_boxes) * self.distill_box
            else:
                # Intermediate feature – distill features
                if self.distill_feature_loss:
                    # Teacher feat already on student device
                    student_feat = self.projector[i](student_feat) if student_feat.ndim == 4 else student_feat

                    if feature_main_masking:
                        fg_mask = feature_fg_masks[feature_level_idx]
                        feature_level_idx += 1
                        _, _, h, w = student_feat.shape
                        if fg_mask.any():
                            student_feat = student_feat.permute(0, 2, 3, 1).reshape(student_feat.shape[0], h * w, -1)[fg_mask]
                            teacher_feat = teacher_feat.permute(0, 2, 3, 1).reshape(teacher_feat.shape[0], h * w, -1)[fg_mask]
                        else:
                            continue

                    if self.distill_feature_loss in {"sl2", "scosine"}:
                        loss_distill_feature += self._feature_kd_loss(
                            student_feat, teacher_feat, feat_idx=i, teacher_scores=teacher_scores
                        ) * feature_weight
                    else:
                        loss_distill_feature += self._feature_kd_loss(
                            student_feat, teacher_feat, feat_idx=i
                        ) * feature_weight

        loss_distill = loss_distill_cls + loss_distill_box + loss_distill_feature
        loss_distill_detach = loss_distill.detach()

        # total = [student_box, student_cls, student_dfl, teacher_box, teacher_cls, teacher_dfl, distill]
        total_loss = torch.cat([student_loss, teacher_loss, loss_distill])
        total_loss_detach = torch.cat([student_loss_detach, teacher_loss_detach, loss_distill_detach])
        return total_loss, total_loss_detach

    # ------------------------------------------------------------------
    # Teacher score computation (for sl2 / scosine)
    # ------------------------------------------------------------------

    def _compute_teacher_scores(self, teacher_feats, target_device=None):
        """Compute teacher attention scores for score-weighted distillation."""
        student_model = self._student()
        score_type = student_model.args.sl2_score
        head_feat = teacher_feats[-1]
        head_feat = self._to_device(head_feat, target_device or self.student_device)
        head_feat = self._detach(head_feat)

        if score_type == "one2one":
            scores = self._decouple(head_feat, branch="one2one")["scores"]
            parts = torch.split(scores, [6400, 1600, 400], dim=-1)
            return tuple(p.sigmoid().max(dim=1, keepdim=True).values for p in parts)
        elif score_type == "one2many":
            scores = self._decouple(head_feat, branch="one2many")["scores"]
            parts = torch.split(scores, [6400, 1600, 400], dim=-1)
            return tuple(p.sigmoid().max(dim=1, keepdim=True).values for p in parts)
        elif score_type == "avg":
            s1 = self._decouple(head_feat, branch="one2many")["scores"]
            s2 = self._decouple(head_feat, branch="one2one")["scores"]
            scores = (s1 + s2) / 2
            parts = torch.split(scores, [6400, 1600, 400], dim=-1)
            return tuple(p.sigmoid().max(dim=1, keepdim=True).values for p in parts)
        elif score_type == "o2m":
            o2m = student_model.criterion.o2m
            o2o = student_model.criterion.o2o
            s1 = self._decouple(head_feat, branch="one2many")["scores"]
            s2 = self._decouple(head_feat, branch="one2one")["scores"]
            scores = s1 * o2m + s2 * o2o
            parts = torch.split(scores, [6400, 1600, 400], dim=-1)
            return tuple(p.sigmoid().max(dim=1, keepdim=True).values for p in parts)
        elif score_type == "local_max":
            feats = self._decouple(head_feat, branch="one2many")["scores"]
            parts = torch.split(feats, [6400, 1600, 400], dim=-1)
            sigs = tuple(p.sigmoid() for p in parts)
            ks = (7, 5, 3)
            return tuple(self._keep_local_max(s, k) for s, k in zip(sigs, ks))
        else:
            raise ValueError(f"Unknown sl2_score type: {score_type}")

    # ------------------------------------------------------------------
    # Loss functions (same as DistillationModel)
    # ------------------------------------------------------------------

    @staticmethod
    def _keep_local_max(x, kernel_size=3):
        pooled = F.max_pool2d(x, kernel_size=kernel_size, stride=1, padding=kernel_size // 2)
        mask = x == pooled
        out = (x * mask).max(dim=1, keepdim=True).values
        return out

    def _loss_kl(self, student_logits, teacher_logits, temperature=5.0):
        soft_targets = F.softmax(teacher_logits / temperature, dim=1)
        student_soft = F.log_softmax(student_logits / temperature, dim=1)
        return F.kl_div(student_soft, soft_targets, reduction="mean") * (temperature ** 2)

    def _loss_cosine(self, a, b):
        if a.ndim == 4:
            a = a.flatten(2).permute(0, 2, 1)
        if b.ndim == 4:
            b = b.flatten(2).permute(0, 2, 1)
        a = F.normalize(a, p=2, dim=-1)
        b = F.normalize(b, p=2, dim=-1)
        return (1 - F.cosine_similarity(a, b, dim=-1)).mean()

    def _loss_mgd(self, student_feat, teacher_feat, lambda_mgd=0.65, feat_idx=0):
        N, C, H, W = teacher_feat.shape
        mat = torch.rand((N, 1, H, W), device=student_feat.device)
        mat = torch.where(mat > 1 - lambda_mgd, 0, 1).float()
        masked = student_feat * mat
        new_fea = self.generation[feat_idx](masked)
        return nn.MSELoss(reduction="sum")(new_fea, teacher_feat) / N

    def _loss_cwd(self, student_feat, teacher_feat, feat_idx=0, temperature=1.0):
        student_feat = self.norm[feat_idx](student_feat)
        teacher_feat = self.norm[feat_idx](teacher_feat)
        N, C, H, W = teacher_feat.shape
        soft_t = F.softmax(teacher_feat.view(-1, W * H) / temperature, dim=1)
        logsoftmax = nn.LogSoftmax(dim=1)
        cost = torch.sum(
            soft_t * logsoftmax(teacher_feat.view(-1, W * H) / temperature)
            - soft_t * logsoftmax(student_feat.view(-1, W * H) / temperature)
        ) * (temperature ** 2)
        return cost / (N * C)

    def _loss_sl2(self, student_feat, teacher_feat, feat_idx=0, teacher_scores=None):
        score = teacher_scores[feat_idx]
        N, C, H, W = student_feat.shape
        s = student_feat.view(N, C, -1)
        t = teacher_feat.view(N, C, -1)
        loss = F.mse_loss(s, t, reduction="none") * score
        return loss.sum() / (score.sum() * C + 1e-9)

    def _loss_s_cosine(self, student_feat, teacher_feat, feat_idx=0, teacher_scores=None):
        score = teacher_scores[feat_idx]
        B, C, H, W = student_feat.shape
        s = F.normalize(student_feat, p=2, dim=1).view(B, C, -1)
        t = F.normalize(teacher_feat, p=2, dim=1).view(B, C, -1)
        cos = (s * t).sum(dim=1, keepdim=True)
        loss = (1.0 - cos) * score
        return loss.sum() / (score.sum() + 1e-9)

    def _bbox_decode(self, anchor_points, pred_dist):
        c = pred_dist.shape[-1]
        if c % 4 != 0:
            raise ValueError(f"Expected box channels divisible by 4, got {c}")
        if c > 4:
            reg_max = c // 4
            proj = torch.arange(reg_max, dtype=pred_dist.dtype, device=pred_dist.device)
            pred_dist = pred_dist.view(*pred_dist.shape[:-1], 4, reg_max).softmax(-1).matmul(proj)
        return dist2bbox(pred_dist, anchor_points, xywh=False)

    def _loss_iou(self, s_boxes, t_boxes, anchor_points):
        if s_boxes.numel() == 0:
            return s_boxes.sum() * 0.0
        s_dec = self._bbox_decode(anchor_points, s_boxes)
        t_dec = self._bbox_decode(anchor_points, t_boxes)
        return (1.0 - bbox_iou(s_dec, t_dec, xywh=False, CIoU=True)).mean()

    def _loss_siou(self, s_boxes, t_boxes, anchor_points, t_scores):
        if s_boxes.numel() == 0:
            return s_boxes.sum() * 0.0
        if t_scores is None:
            raise ValueError("teacher_score_weight required for 'siou'.")
        s_dec = self._bbox_decode(anchor_points, s_boxes)
        t_dec = self._bbox_decode(anchor_points, t_boxes)
        ciou = bbox_iou(s_dec, t_dec, xywh=False, CIoU=True)
        base = 1.0 - ciou
        return (base * t_scores).sum() / (t_scores.sum() + 1e-9)

    def _loss_s_sigmoid(self, student_logits, teacher_logits):
        nc = student_logits.shape[1]
        loss = F.binary_cross_entropy_with_logits(student_logits, torch.sigmoid(teacher_logits), reduction="none")
        score = teacher_logits.sigmoid().max(dim=1, keepdim=True).values
        return (loss * score).sum() / (score.sum() * nc + 1e-9)

    # ------------------------------------------------------------------
    # Dispatch helpers
    # ------------------------------------------------------------------

    def _cls_kd_loss(self, s_logits, t_logits, temperature=5.0):
        from ultralytics.utils.torch_utils import autocast
        with autocast(enabled=False):
            s_logits, t_logits = s_logits.float(), t_logits.float()
            if self.distill_cls_loss == "softmax":
                return self._loss_kl(s_logits, t_logits, temperature)
            elif self.distill_cls_loss == "sigmoid":
                return F.binary_cross_entropy_with_logits(
                    s_logits / temperature, torch.sigmoid(t_logits / temperature), reduction="mean"
                ) * (temperature ** 2)
            elif self.distill_cls_loss == "s_sigmoid":
                return self._loss_s_sigmoid(s_logits, t_logits)
            else:
                raise ValueError(f"Unknown cls distill loss: {self.distill_cls_loss}")

    def _box_kd_loss(self, s_boxes, t_boxes, anchor_points=None, t_scores=None):
        if self.distill_box_loss == "cos":
            return self._loss_cosine(s_boxes, t_boxes)
        elif self.distill_box_loss == "l1":
            return F.l1_loss(s_boxes, t_boxes)
        elif self.distill_box_loss == "l2":
            return F.mse_loss(s_boxes, t_boxes)
        elif self.distill_box_loss == "iou":
            return self._loss_iou(s_boxes, t_boxes, anchor_points)
        elif self.distill_box_loss == "siou":
            return self._loss_siou(s_boxes, t_boxes, anchor_points, t_scores)
        else:
            raise ValueError(f"Unknown box distill loss: {self.distill_box_loss}")

    def _feature_kd_loss(self, s_feat, t_feat, feat_idx=0, teacher_scores=None):
        if self.distill_feature_loss == "cos":
            return self._loss_cosine(s_feat, t_feat)
        elif self.distill_feature_loss == "l1":
            return F.l1_loss(s_feat, t_feat)
        elif self.distill_feature_loss == "l2":
            return F.mse_loss(s_feat, t_feat)
        elif self.distill_feature_loss == "mgd":
            return self._loss_mgd(s_feat, t_feat, feat_idx=feat_idx)
        elif self.distill_feature_loss == "cwd":
            return self._loss_cwd(s_feat, t_feat, feat_idx=feat_idx)
        elif self.distill_feature_loss == "sl2":
            return self._loss_sl2(s_feat, t_feat, feat_idx=feat_idx, teacher_scores=teacher_scores)
        elif self.distill_feature_loss == "scosine":
            return self._loss_s_cosine(s_feat, t_feat, feat_idx=feat_idx, teacher_scores=teacher_scores)
        else:
            raise ValueError(f"Unknown feature distill loss: {self.distill_feature_loss}")

    # ------------------------------------------------------------------
    # Utility / property methods
    # ------------------------------------------------------------------

    def _decouple(self, preds, shape_check=False, branch="one2one"):
        """Decouple outputs for teacher/student models."""
        if isinstance(preds, tuple):
            preds = preds[1]
        if isinstance(preds, dict):
            if branch in preds:
                preds = preds[branch]
            if shape_check:
                preds = preds["boxes"]
        return preds

    @property
    def criterion(self):
        """Get criterion from student model."""
        return self._student().criterion

    @criterion.setter
    def criterion(self, value):
        """Set criterion for both models."""
        self._student().criterion = value
        self._teacher().criterion = value

    @property
    def teacher_criterion(self):
        """Get criterion from teacher model."""
        return self._teacher().criterion

    @teacher_criterion.setter
    def teacher_criterion(self, value):
        """Set criterion for teacher model."""
        self._teacher().criterion = value

    @property
    def end2end(self):
        return getattr(self._student(), "end2end", False)

    @end2end.setter
    def end2end(self, value):
        self._student().end2end = value
        self._teacher().end2end = value

    def set_head_attr(self, **kwargs):
        """Forward head-attribute updates to both models."""
        student = self._student()
        teacher = self._teacher()
        if hasattr(student, "set_head_attr"):
            student.set_head_attr(**kwargs)
        if hasattr(teacher, "set_head_attr"):
            teacher.set_head_attr(**kwargs)

    def fuse(self, verbose=True):
        self._student().fuse(verbose)
        return self

    def load_from_module(self, weights, strict=False):
        """Load co-training weights from a checkpoint dict or module."""
        module = weights["model"] if isinstance(weights, dict) else weights
        if not isinstance(module, nn.Module):
            raise TypeError(f"Expected nn.Module or checkpoint dict, got {type(weights).__name__}")
        state_dict = module.float().state_dict()
        incompatible = self.load_state_dict(state_dict, strict=strict)
        if strict or (not incompatible.missing_keys and not incompatible.unexpected_keys):
            return incompatible

        remapped = {}
        for k, v in state_dict.items():
            nk = k
            if nk.startswith("module."):
                nk = nk[len("module."):]
            nk = nk.replace("student_model.module.", "student_model.")
            nk = nk.replace("teacher_model.module.", "teacher_model.")
            remapped[nk] = v
        incompatible = self.load_state_dict(remapped, strict=False)
        return incompatible
