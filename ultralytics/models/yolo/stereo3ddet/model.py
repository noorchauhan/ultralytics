# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import torch

from ultralytics.nn.modules.block import StereoCostVolume
from ultralytics.nn.tasks import DetectionModel


class Stereo3DDetModel(DetectionModel):
    """Stereo 3D Detection model — 6 channel input (stereo pair).

    Supports two backbone modes:
    - groups=2 stem (default): 6ch input with groups=2 in first conv layers
    - siamese (siamese: true in YAML): standard 3ch backbone runs on L/R images
      separately via batch trick, features split at tap point for cost volume
    """

    def __init__(self, cfg, ch=None, nc=None, verbose=True):
        # Load YAML to check for siamese flag before building the model
        from ultralytics.nn.tasks import yaml_model_load

        yaml_cfg = cfg if isinstance(cfg, dict) else yaml_model_load(cfg)
        siamese = yaml_cfg.get("siamese", False)

        # Siamese uses 3ch backbone; disable siamese during __init__ so stride
        # computation uses standard forward (model not fully set up yet)
        self._siamese = False
        if siamese:
            ch = 3  # force 3ch backbone regardless of what trainer passes
        elif ch is None:
            ch = 6

        super().__init__(cfg=cfg, ch=ch, nc=nc, verbose=verbose)
        self.task = "stereo3ddet"
        self.end2end = False  # stereo uses custom geometric postprocessing, not NMS-free

        # Now enable siamese mode and find tap/cv layer indices
        if siamese:
            for m in self.model:
                if isinstance(m, StereoCostVolume):
                    self._tap_layer = m.f  # backbone layer whose output feeds cost volume
                    self._cv_layer = m.i  # StereoCostVolume layer index
                    self._siamese = True
                    break

        # Apply depth_mode from YAML (prune unused aux branches)
        depth_mode = (self.yaml or {}).get("training", {}).get("depth_mode", "both")
        if depth_mode != "both":
            from ultralytics.models.yolo.stereo3ddet.head_yolo11 import Stereo3DDetHeadYOLO11

            head = self.model[-1]
            if isinstance(head, Stereo3DDetHeadYOLO11):
                head.set_depth_mode(depth_mode)

    def _predict_once(self, x, profile=False, visualize=False, embed=None):
        """Forward pass with siamese batch trick for stereo input.

        For 6ch stereo input (siamese mode):
        1. Split into left [B,3] and right [B,3], stack as [2B,3,H,W]
        2. Run backbone layers 0..tap_layer on [2B,3] (shared weights)
        3. Split: left_feat [B,C], right_feat [B,C]
        4. Continue layers tap_layer+1..end with left-only features
        5. StereoCostVolume receives (left_feat, right_feat) tuple

        For 3ch input (stride computation) or non-siamese: standard forward.
        """
        if not self._siamese or x.shape[1] != 6:
            return super()._predict_once(x, profile, visualize, embed)

        B = x.shape[0]
        left = x[:, :3]
        right = x[:, 3:]
        x = torch.cat([left, right], dim=0)  # [2B, 3, H, W]

        y, dt, embeddings = [], [], []
        embed = frozenset(embed) if embed is not None else {-1}
        max_idx = max(embed)
        right_tap = None  # right features at tap point

        for m in self.model:
            if m.f != -1:
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]

            # At StereoCostVolume layer, pass (left_tap, right_tap) tuple
            if m.i == self._cv_layer:
                x = m((y[self._tap_layer][:B], right_tap))
            else:
                if profile:
                    self._profile_one_layer(m, x, dt)
                x = m(x)

            # At tap layer, split L/R and continue with left-only
            if m.i == self._tap_layer:
                right_tap = x[B:]  # save right features for cost volume
                x = x[:B]  # continue with left-only

            y.append(x if m.i in self.save else None)
            if visualize:
                from ultralytics.utils import feature_visualization

                feature_visualization(x, m.type, m.i, save_dir=visualize)
            if m.i in embed:
                embeddings.append(torch.nn.functional.adaptive_avg_pool2d(x, (1, 1)).squeeze(-1).squeeze(-1))
                if m.i == max_idx:
                    return torch.unbind(torch.cat(embeddings, 1), dim=0)

        return x

    def init_criterion(self):
        """Initialize the loss criterion."""
        from ultralytics.models.yolo.stereo3ddet.loss_yolo11 import Stereo3DDetLossYOLO11

        aux_w = None
        use_bbox_loss = True
        if hasattr(self, "yaml") and self.yaml is not None:
            training_config = self.yaml.get("training", {})
            if training_config:
                if "loss_weights" in training_config:
                    aux_w = training_config["loss_weights"]
                if "use_bbox_loss" in training_config:
                    use_bbox_loss = bool(training_config["use_bbox_loss"])

        return Stereo3DDetLossYOLO11(self, loss_weights=aux_w, use_bbox_loss=use_bbox_loss)
