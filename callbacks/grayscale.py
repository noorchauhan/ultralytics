"""Callback to inject RandomGrayscale into classification training transforms.

Usage:
    from callbacks import grayscale
    model.add_callback("on_train_start", grayscale.override(0.2))
"""


def override(p=0.2):
    """Return on_train_start callback to add RandomGrayscale to train transforms.

    Args:
        p (float): Probability of converting image to grayscale.
    """

    def callback(trainer):
        import torchvision.transforms as T

        transforms = list(trainer.train_loader.dataset.torch_transforms.transforms)
        for i, t in enumerate(transforms):
            if isinstance(t, T.ToTensor):
                transforms.insert(i, T.RandomGrayscale(p=p))
                break
        trainer.train_loader.dataset.torch_transforms = T.Compose(transforms)

    return callback
