"""Callback to override MuSGD muon weight at train start.

Usage:
    from callbacks import muon_w
    model.add_callback("on_train_start", muon_w.override(0.1))
"""


def override(scale=0.1):
    """Return on_train_start callback to set optimizer.muon weight.

    Args:
        scale (float): Muon weight value.
    """

    def callback(trainer):
        if hasattr(trainer.optimizer, "muon"):
            trainer.optimizer.muon = scale

    return callback
