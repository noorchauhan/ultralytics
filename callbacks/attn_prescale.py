"""Callback to patch Attention modules with pre-scaled Q for fp16 stability.

Post-scaling (q @ k) * scale can overflow fp16 max (65504) when Q/K magnitudes
grow during training. Pre-scaling (q * scale) @ k keeps values in range.
Same trick used by Apple MobileCLIP (mci.py:141).

Usage:
    from callbacks import attn_prescale
    model.add_callback("on_train_start", attn_prescale.override())
"""

import types


def override():
    """Return on_train_start callback to patch Attention forward with pre-scaled Q."""

    def callback(trainer):
        from ultralytics.nn.modules.block import AAttn, Attention

        def _attn_forward(self, x):
            B, C, H, W = x.shape
            N = H * W
            qkv = self.qkv(x)
            q, k, v = qkv.view(B, self.num_heads, self.key_dim * 2 + self.head_dim, N).split(
                [self.key_dim, self.key_dim, self.head_dim], dim=2
            )
            attn = ((q * self.scale).transpose(-2, -1) @ k).softmax(dim=-1)
            x = (v @ attn.transpose(-2, -1)).view(B, C, H, W) + self.pe(v.reshape(B, C, H, W))
            return self.proj(x)

        def _aattn_forward(self, x):
            B, C, H, W = x.shape
            N = H * W
            qkv = self.qkv(x).flatten(2).transpose(1, 2)
            if self.area > 1:
                qkv = qkv.reshape(B * self.area, N // self.area, C * 3)
                B, N, _ = qkv.shape
            q, k, v = (
                qkv.view(B, N, self.num_heads, self.head_dim * 3)
                .permute(0, 2, 3, 1)
                .split([self.head_dim, self.head_dim, self.head_dim], dim=2)
            )
            attn = ((q * (self.head_dim**-0.5)).transpose(-2, -1) @ k).softmax(dim=-1)
            x = v @ attn.transpose(-2, -1)
            x = x.permute(0, 3, 1, 2)
            v = v.permute(0, 3, 1, 2)
            if self.area > 1:
                x = x.reshape(B // self.area, N * self.area, C)
                v = v.reshape(B // self.area, N * self.area, C)
                B, N, _ = x.shape
            x = x.reshape(B, H, W, C).permute(0, 3, 1, 2).contiguous()
            v = v.reshape(B, H, W, C).permute(0, 3, 1, 2).contiguous()
            x = x + self.pe(v)
            return self.proj(x)

        for module in trainer.model.modules():
            if isinstance(module, Attention):
                module.forward = types.MethodType(_attn_forward, module)
            elif isinstance(module, AAttn):
                module.forward = types.MethodType(_aattn_forward, module)

    return callback
