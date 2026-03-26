"""Callback to add stochastic depth (drop path) to PSABlock residual connections.

Standard regularization for attention models (timm uses 0.1-0.35). Randomly
drops entire residual branches during training, reducing overfitting.

Usage:
    from callbacks import droppath
    model.add_callback("on_train_start", droppath.override(drop_prob=0.1))
"""

import types

import torch


def _drop_path(x, drop_prob, training):
    """Apply stochastic depth by randomly zeroing entire residual branches."""
    if drop_prob == 0.0 or not training:
        return x
    keep = 1 - drop_prob
    mask = torch.floor(torch.rand((x.shape[0],) + (1,) * (x.ndim - 1), dtype=x.dtype, device=x.device) + keep)
    return x * mask / keep


def override(drop_prob=0.1):
    """Return on_train_start callback to patch PSABlock forward with drop path.

    Args:
        drop_prob (float): Probability of dropping each residual branch.
    """

    def callback(trainer):
        from ultralytics.nn.modules.block import PSABlock

        def _forward(self, x):
            x = x + _drop_path(self.attn(x), drop_prob, self.training) if self.add else self.attn(x)
            x = x + _drop_path(self.ffn(x), drop_prob, self.training) if self.add else self.ffn(x)
            return x

        for module in trainer.model.modules():
            if isinstance(module, PSABlock):
                module.forward = types.MethodType(_forward, module)

    return callback
