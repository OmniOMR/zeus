import re
from collections.abc import Iterable

import numpy as np
import tensorflow as tf

from ..data.shuffled_view import ShuffledView
from .architecture_options import ArchitectureOptions
from .inference_options import InferenceOptions
from .token_map import TokenMap
from .training_options import TrainingOptions


def construct_tf_dataset(
    shuffled_view: ShuffledView,
    architecture_options: ArchitectureOptions,
    token_map: TokenMap,
    training_or_inference_options: TrainingOptions | InferenceOptions,
) -> tf.data.Dataset:
    """
    Returns a tf.data.Dataset representation of a ZeusDataset for either
    training or inference, based on the given options object.

    When training, the data is shuffled and augmented.

    When running inference, images may be transformed before
    being fed into the model (i.e. binarized).

    :param architecture_options: Used to get the desired image height.
    :param token_map: Map used to convert LMX tokens to feature indexes
        for the model output layer.
    :param training_or_inference_options: Options to either prepare the
        data for training or for inference.
    """

    # parse out needed values from options
    options = training_or_inference_options
    if isinstance(options, TrainingOptions):
        is_training = True
        is_finetuning = options.is_finetuning
        transformations: list[str] = []
        augmentations: str = options.augmentations
        batch_size: int = options.batch_size
        seed: int = options.seed
        max_image_width: int | None = options.max_image_width
    elif isinstance(options, InferenceOptions):
        is_training = False
        is_finetuning = False
        transformations = options.transformations
        augmentations = ""  # not used
        batch_size = options.batch_size
        seed = 0  # not used
        max_image_width = options.max_image_width

    def generator() -> Iterable[tuple[bytes, np.ndarray]]:
        """
        Emits pairs of (image, lmx token indices), where the image is kept
        in its original binary representation to conserve RAM space.
        Images are decoded only as they are needed for the next batch.
        LMX is however decoded into indexes right away.
        """
        nonlocal is_training, token_map
        for sample in shuffled_view.iter_shuffled_samples():
            yield (
                sample.image,
                np.array(
                    [
                        token_map.token_to_index(
                            token, allow_unknown_tokens=(not is_training or is_finetuning)
                        )
                        for token in sample.lmx.split()
                    ],
                    dtype=np.int32,
                ),
            )

    def prepare_example(
        image_bytes: bytes, token_indexes: np.ndarray
    ) -> tuple[tf.Tensor, np.ndarray]:
        nonlocal architecture_options, transformations, max_image_width
        # The decoded tensor gets a name of its own rather than being written
        # back over the parameter, which is bytes and stays bytes.
        image = tf.image.convert_image_dtype(
            tf.image.decode_image(image_bytes, channels=1, expand_animations=False), tf.float32
        )
        for transformation, *parameters in (part.split(":") for part in transformations):
            if transformation == "threshold":
                # Parsed into new names rather than over the originals, so that
                # `lower` and `upper` are floats throughout instead of starting
                # life as the strings they were spelled with.
                lower_text, upper_text, *rest = parameters
                lower, upper = float(lower_text), float(upper_text)
                smooth = rest.count("smooth")
                if not smooth:
                    image = tf.cast(image >= lower, tf.float32) * tf.cast(
                        image <= upper, tf.float32
                    ) * image + tf.cast(image > upper, tf.float32)
                else:
                    image = tf.clip_by_value((image - lower) / (upper - lower), 0.0, 1.0)
            elif transformation:
                raise ValueError(f"The transformation '{transformation}' is unknown.")
        image = tf.image.resize(
            image,
            size=[architecture_options.height, max_image_width or tf.int32.max],
            preserve_aspect_ratio=True,
            antialias=True,
        )
        return image, token_indexes

    if is_training:
        # Prepare augmentation operations, reused in between batches
        rng = tf.random.Generator.from_seed(seed)
        if match := re.search(r"rotate:([\d.]+)", augmentations):
            rng_rotation = tf.keras.layers.RandomRotation(
                float(match.group(1)) / 360,
                fill_mode="constant",
                interpolation="bilinear",
                seed=seed,
                fill_value=1.0,
            )

    def augment(image, tags):
        nonlocal rng, rng_rotation, augmentations, architecture_options
        for augmentation, *parameters in (part.split(":") for part in augmentations.split(",")):
            if rng.uniform([], 0, 1) >= 0.5:
                continue
            if augmentation == "h":
                a = int(parameters[0])
                image = tf.pad(image, [[0, 0], [a, 0], [0, 0]], constant_values=1.0)[
                    :, rng.uniform([], 0, 2 * a + 1, dtype=tf.int32) :
                ]
            elif augmentation == "v":
                a = int(parameters[0])
                image = tf.pad(image, [[a, a], [0, 0], [0, 0]], constant_values=1.0)[
                    rng.uniform([], 0, 2 * a + 1, dtype=tf.int32) :
                ]
                image = image[: architecture_options.height]
            elif augmentation == "rotate":
                image = rng_rotation(image, training=True)
            elif augmentation == "b":
                lower, upper = map(float, parameters[:2])
                image = tf.clip_by_value(
                    tf.image.adjust_brightness(image, rng.uniform([], lower, upper)), 0.0, 1.0
                )
            elif augmentation == "c":
                lower, upper = map(float, parameters[:2])
                image = tf.clip_by_value(
                    tf.image.adjust_contrast(image, 2 ** rng.uniform([], lower, upper)), 0.0, 1.0
                )
            elif augmentation == "n":
                p = float(parameters[0])
                mask = tf.cast(
                    rng.uniform(tf.shape(image), 0, 1) >= rng.uniform([], 0, p), tf.float32
                )
                image = mask * image + (1 - mask) * (1 - image)
            elif augmentation == "en3":
                p = float(parameters[0])
                mask = tf.nn.avg_pool2d(image[tf.newaxis], 3, 1, padding="SAME")[0]
                mask = tf.cast((mask <= 0.1) | (mask >= 0.9), tf.float32)
                mask = mask + (1 - mask) * tf.cast(
                    rng.uniform(tf.shape(mask), 0, 1) >= rng.uniform([], 0, p), tf.float32
                )
                image = mask * image + (1 - mask) * (1 - image)
            elif augmentation == "de":
                d = rng.uniform([], -np.pi / 2, np.pi / 2)
                x, y = tf.cos(d), 0.5 * tf.sin(d)
                moved = tf.raw_ops.ImageProjectiveTransformV3(
                    images=image[tf.newaxis],
                    transforms=[[1.0, 0.0, x, 0.0, 1.0, y, 0.0, 0.0]],
                    output_shape=tf.shape(image)[:2],
                    fill_value=1.0,
                    interpolation="BILINEAR",
                )[0]
                if rng.uniform([], 0, 1) >= 0.5:
                    image = tf.math.maximum(image, moved)
                else:
                    image = tf.clip_by_value(image + moved - 1, 0.0, 1.0)
            elif augmentation:
                raise ValueError(f"The augmentation '{augmentation}' is unknown.")
        return image, tags

    # === The dataset pipeline ===

    dataset = tf.data.Dataset.from_generator(
        generator,
        output_signature=(
            tf.TensorSpec(shape=(), dtype=tf.string),
            tf.TensorSpec(shape=(None,), dtype=tf.int32),
        ),
    )
    dataset = dataset.cache()
    dataset = dataset.apply(
        tf.data.experimental.assert_cardinality(expected_cardinality=sum(1 for _ in dataset))
    )
    if is_training:
        dataset = dataset.shuffle(5_000, seed=seed)
    dataset = dataset.map(prepare_example, num_parallel_calls=tf.data.AUTOTUNE)
    if is_training and augmentations:
        dataset = dataset.map(augment, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.ragged_batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset
