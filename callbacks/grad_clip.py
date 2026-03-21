"""Callback to replace optimizer_step with grad-clipped version.

Usage:
    from callbacks import grad_clip
    model.add_callback("on_train_start", grad_clip.override(1.0))
"""

import types

import torch


def override(max_norm=1.0):
    """Return on_train_start callback to replace optimizer_step with grad clipping.

    Args:
        max_norm (float): Max gradient norm for clipping.
    """

    def callback(trainer):
        def _step(self):
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=max_norm)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad()
            if self.ema:
                self.ema.update(self.model)

        trainer.optimizer_step = types.MethodType(_step, trainer)

    return callback
