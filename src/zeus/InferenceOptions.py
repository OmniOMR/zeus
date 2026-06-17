from dataclasses import dataclass, field


@dataclass
class InferenceOptions:
    """Options that constrain the way the model does inference"""
    
    batch_size: int = 64
    """How many samples to use per inference batch"""

    transformations: list[str] = field(default_factory=list)
    """
    List of image transformations to apply to each sample,
    may be used to binarize grayscale images.
    """

    max_prediction_length: int = 700
    """Maximum length of a predicted sequence"""

    max_image_width: int | None = None
    """
    Maximum image width to feed into the model, None means unlimited.
    Larger images are scaled-down to fit into this limit.
    """
