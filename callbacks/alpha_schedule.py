"""Callback to decay CLIPDistillationLoss alpha on a cosine schedule.

Alpha controls the CE vs KL balance: 0 = pure CE, 1 = pure distillation.
Cosine decay from start to end over total epochs lets the model learn alignment
early and refine classification later.

Usage:
    from callbacks import alpha_schedule
    model.add_callback("on_train_epoch_start", alpha_schedule.override(start=1.0, end=0.1))
"""

import math


def override(start=1.0, end=0.1):
    """Return on_train_epoch_start callback to update criterion alpha via cosine decay.

    Args:
        start (float): Initial alpha value at epoch 0.
        end (float): Final alpha value at last epoch.
    """

    def callback(trainer):
        model = trainer.model
        if hasattr(model, "module"):
            model = model.module
        if not hasattr(model, "criterion") or not hasattr(model.criterion, "alpha"):
            return
        epoch = trainer.epoch
        epochs = trainer.epochs
        alpha = end + 0.5 * (start - end) * (1 + math.cos(math.pi * epoch / epochs))
        model.criterion.alpha = alpha

    return callback
