from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import yaml


@dataclass
class TrainingOptions:
    """Options that constrain the way the model is trained"""

    epochs: int | None
    """How many epochs to train for. None when training stops early instead,
    see `early_stopping_patience`."""

    evaluation_from: int
    """
    Start evaluating only after a given epoch
    (including this one) (1-based epoch index)
    """

    evaluation_each: int
    """Run evaluation each this number of epochs."""

    is_finetuning: bool
    """False when training a new model from scratch, true when training
    an already once trained model."""

    augmentations: str = "h:8"
    """List of image augmentations to randomly choose from during training"""

    batch_size: int = 64
    """How many samples to use per training batch"""

    learning_rate: float = 1e-3
    """Initial learning rate"""

    lr_decay: Literal["none"] | Literal["cos"] = "cos"
    """Learning rate decay type"""

    max_training_length: int = 500
    """Maximum length of a training sequence. Used to determine the size of
    tensors to allocate for the ragged sequences."""

    max_image_width: int | None = 1500
    """
    Maximum image width to feed into the model, None means unlimited.
    Larger images are scaled-down to fit into this limit.
    Defaults to 1500px, which is wider than most OmniOMR and olimpic samples
    when height-normalized.
    """

    seed: int = 42
    """Random seed to be used for augmentations and shuffling"""

    early_stopping_patience: int | None = None
    """When set, training has no fixed length: it runs until SER on the early
    stopping dataset has not reached a new minimum for this many epochs.
    Exclusive with `epochs` and with learning rate decay."""

    def __post_init__(self):
        if (self.epochs is None) == (self.early_stopping_patience is None):
            raise ValueError(
                "Training needs exactly one of a fixed number of epochs "
                + "or an early stopping patience."
            )
        if self.early_stopping_patience is not None:
            if self.early_stopping_patience < 1:
                raise ValueError("Early stopping patience must be at least 1.")
            # Cosine decay is a schedule over a known number of steps, and an
            # early-stopped run does not know its length. A decaying rate would
            # also make the SER curve flatten for reasons other than the model
            # having stopped improving.
            if self.lr_decay != "none":
                raise ValueError("Early stopping requires a fixed learning rate, lr_decay='none'.")

    @property
    def is_early_stopping(self) -> bool:
        """Whether training runs until it stops improving, not for fixed epochs."""
        return self.early_stopping_patience is not None

    def write_to_yaml_file(self, file_path: Path):
        """Writes training options into a yaml file"""
        yaml_data: dict = asdict(self)
        with open(file_path, "w") as file:
            yaml.dump(yaml_data, file)
