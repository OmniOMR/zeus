"""Turning an encoded image into the tensor the encoder expects.

This is the one step that training, evaluation and inference must agree on
exactly. A model trained on images normalized one way and fed images
normalized another way does not fail — it just predicts badly, which is the
kind of bug that takes a week to find. So there is one implementation of it,
here, and every path calls it.
"""

import tensorflow as tf


def preprocess_image(
    image_bytes: tf.Tensor,
    height: int,
    transformations: list[str],
    max_image_width: int | None,
) -> tf.Tensor:
    """Decode one encoded image and normalize it for the encoder.

    :param image_bytes: The image file's bytes, as a scalar string tensor.
        PNG and JPEG both work.
    :param height: The height the architecture expects. The image is scaled to
        exactly this, and its width follows from its aspect ratio.
    :param transformations: Transformations to apply before scaling, such as
        `threshold:0.3:0.7` to binarize a grayscale scan. Empty for training,
        where augmentation happens separately and afterwards.
    :param max_image_width: Width beyond which a height-normalized image is
        scaled down further, trading resolution for a bounded tensor. `None`
        leaves the width unbounded.
    :returns: A `[height, width, 1]` float tensor with values in `[0, 1]`.
    """
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

    # The height is non-negotiable: the encoder's input layer declares it
    # statically, and an image even one pixel short fails deep inside the graph
    # with an unreadable reshape error rather than anything that names the
    # cause. So the height is set exactly and only the width gives.
    #
    # This used to be a single resize with `preserve_aspect_ratio=True` and a
    # size of `[height, max_image_width]`, which is a fit-inside-the-box: for
    # any staff wider than the cap the *width* bound first and the height came
    # out short, breaking the model. Two thirds of the dolores samples are that
    # wide, so training and evaluation on it could not run at all.
    #
    # Aspect ratio, exact height and a bounded width cannot all three hold, and
    # aspect ratio is the one to give up. Vertical resolution is where pitch
    # lives — the reader is measuring which line or space a notehead sits on —
    # while horizontal compression only crowds the notes together, which a
    # left-to-right recurrent reader tolerates well.
    original_shape = tf.shape(image)
    width_at_full_height = tf.cast(
        tf.round(
            tf.cast(original_shape[1] * height, tf.float32) / tf.cast(original_shape[0], tf.float32)
        ),
        tf.int32,
    )
    width = width_at_full_height
    if max_image_width is not None:
        width = tf.minimum(width, max_image_width)

    # At least one pixel wide, so that a degenerate input cannot produce an
    # empty tensor that fails somewhere less obvious.
    width = tf.maximum(width, 1)

    return tf.image.resize(image, size=[height, width], antialias=True)
