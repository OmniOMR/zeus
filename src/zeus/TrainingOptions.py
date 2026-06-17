from dataclasses import dataclass, asdict
from typing import Literal
from pathlib import Path
import yaml


@dataclass
class TrainingOptions:
    """Options that constrain the way the model is trained"""

    epochs: int
    """How many epochs to train for."""

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

    max_image_width: int | None = None
    """
    Maximum image width to feed into the model, None means unlimited.
    Larger images are scaled-down to fit into this limit.
    """

    seed: int = 42
    """Random seed to be used for augmentations and shuffling"""

    def write_to_yaml_file(self, file_path: Path):
        """Writes training options into a yaml file"""
        yaml_data: dict = asdict(self)
        with open(file_path, "w") as file:
            yaml.dump(yaml_data, file)
