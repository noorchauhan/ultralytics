from ultralytics.utils.torch_utils import copy_attr, smart_inference_mode
from ultralytics.utils.checks import check_requirements
from .tasks import load_checkpoint
import torch.nn.functional as F
from torch import nn
import torch


class DistillationModel(nn.Module):
    """Distillation model wrapper.
    Currently only supports feature-based distillation with a single feature index on YOLO models.
    """

    def __init__(self, teacher_model: str | nn.Module, student_model: nn.Module, feats_idx: int):
        """Initialize DistillationModel."""
        super().__init__()
        if isinstance(feats_idx, int):
            feats_idx = [feats_idx]
        if isinstance(teacher_model, str):
            self.is_timm, teacher_model = self.get_teacher_model(teacher_model)
        device = next(student_model.parameters()).device
        # self.teacher_model = teacher_model.to(device).eval()
        self.teacher_model = teacher_model.to(device).train()
        if self.is_timm:
            self.mean = torch.tensor([0.485, 0.456, 0.406]).view(-1, 1, 1).to(device)
            self.std = torch.tensor([0.229, 0.224, 0.225]).view(-1, 1, 1).to(device)
        else:
            self.mean = torch.tensor([0, 0, 0]).view(-1, 1, 1).to(device)
            self.std = torch.tensor([1, 1, 1]).view(-1, 1, 1).to(device)

        for v in self.teacher_model.parameters():
            v.requires_grad = False
        self.student_model = student_model
        self.feats_idx = feats_idx
        # get the feature dimensions
        with torch.inference_mode():
            dummy_input = torch.zeros(1, 3, 256, 256).to(device)
            teacher_output = self.get_teacher_output(dummy_input)
            student_output = student_model(dummy_input, embed=feats_idx, direct_return=True)
            assert len(teacher_output) == len(student_output), "Feature dimensions must match in length."
        self.projector = nn.ModuleList(
            nn.Linear(student_dim, teacher_dim) if student_dim != teacher_dim else nn.Identity()
            for student_out, teacher_out in zip(student_output, teacher_output)
            for student_dim, teacher_dim in [
                (
                    student_out.shape[1],
                    teacher_out[1].shape[1] if isinstance(teacher_out, tuple) else teacher_out.shape[1],
                )
            ]
        )
        copy_attr(self, student_model)

    def loss_kl(self, teacher_logits, student_logits, temperature: float = 5):
        """The KL divergence loss for knowledge distillation."""
        soft_targets = F.softmax(teacher_logits / temperature, dim=1)
        student_soft_logits = F.log_softmax(student_logits / temperature, dim=1)

        # Distillation loss (Kullback-Leibler divergence)
        distillation_loss = F.kl_div(student_soft_logits, soft_targets, reduction="batchmean") * (temperature**2)
        return distillation_loss

    def get_teacher_model(self, model_name: str):
        if "dinov3" in model_name:
            check_requirements("timm>=1.0.26")
            import timm

            model = timm.create_model(model_name, pretrained=True, global_pool="token")
            is_timm = True
        else:
            model = load_checkpoint(model_name)[0]
            is_timm = False
        return is_timm, model

    @smart_inference_mode()
    def get_teacher_output(self, input: torch.Tensor):
        if self.is_timm:
            input = (input - self.mean.to(input.dtype)) / self.std.to(input.dtype)
            _, teacher_output = self.teacher_model.forward_intermediates(
                input,
                indices=[11],  # hardcoded to one layer for now
                output_fmt="NCHW",
                return_prefix_tokens=False,
                norm=True,  # Apply layer norm
            )
        else:
            teacher_output = self.teacher_model(input, embed=self.feats_idx, direct_return=True)
        return teacher_output

    def forward(self, x, *args, **kwargs):
        """Forward pass through the student model."""
        if isinstance(x, dict):  # for cases of training and validating while training.
            return self.loss(x, *args, **kwargs)
        return self.student_model.predict(x, *args, **kwargs)
        # y = []  # outputs
        # sx = x.clone()
        # for i, m in enumerate(self.student_model.model):
        #     sx = m(sx)  # run
        #     if i == 10:
        #         sx = self.projector[0](sx.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        #         break
        # for i, m in enumerate(self.teacher_model.model):
        #     if m.f != -1:  # if not from previous layer
        #         x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers
        #     x = m(x)  # run
        #     if i == 10:
        #         y.append(sx)
        #     else:
        #         y.append(x if m.i in self.save else None)  # save output
        # return x

    def loss(self, batch, preds=None):
        """
        Compute loss.
        Args:
            batch (dict): Batch to compute loss on.
            preds (torch.Tensor | list[torch.Tensor], optional): Predictions.
        """
        with torch.inference_mode():
            teacher_feats = self.get_teacher_output(batch["img"])
        preds, feats = self.student_model(batch["img"], return_feats=True)
        loss_distill = torch.zeros(1, device=batch["img"].device)
        for i, feat_idx in enumerate(self.feats_idx):
            # handle head ouput
            teacher_feat = teacher_feats[i][1] if isinstance(teacher_feats[i], tuple) else teacher_feats[i]
            feat = feats[feat_idx][1] if isinstance(feats[feat_idx], tuple) else feats[feat_idx]
            student_feat = self.projector[i](feat.permute(0, 2, 3, 1)).permute(0, 3, 1, 2) if feat.ndim == 4 else feat
            loss_distill += self.loss_cosine(teacher_feat, student_feat) * self.student_model.args.dis
            # loss_distill += self.loss_kl(teacher_feat, student_feat) * self.student_model.args.dis

        regular_loss, regular_loss_detach = self.student_model.loss(batch, preds)
        return torch.cat([regular_loss, loss_distill]), torch.cat([regular_loss_detach, loss_distill.detach()])

    def loss_cosine(self, teacher_feat, student_feat):
        """Compute cosine similarity loss between teacher and student features."""
        if teacher_feat.shape[2:] != student_feat[2:]:
            student_feat = F.interpolate(student_feat, teacher_feat.shape[2:])
        student_feat = F.normalize(student_feat.flatten(2).permute(0, 2, 1), p=2, dim=-1)
        teacher_feat = F.normalize(teacher_feat.flatten(2).permute(0, 2, 1), p=2, dim=-1)
        cos_sim = F.cosine_similarity(student_feat, teacher_feat, dim=-1)
        loss = (1 - cos_sim).mean()
        return loss

    @property
    def criterion(self):
        """Get the criterion from the student model."""
        return self.student_model.criterion

    def fuse(self, verbose: bool = True):
        """Fuse model layers for inference speedup."""
        self.student_model.fuse(verbose)
        return self
