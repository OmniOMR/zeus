"""Tests for image preprocessing — the step training and inference must share.

The encoder declares its input height statically, so an image that comes out
even one pixel short does not fail with anything that names the cause: it fails
deep in the graph with a reshape mismatch. These tests pin the shape contract
so that cannot come back.
"""

import numpy as np
import pytest
import tensorflow as tf

from zeus.model.preprocess_image import preprocess_image

HEIGHT = 96


def an_image(width: int, height: int) -> tf.Tensor:
    """A PNG of the given size, as encoded bytes."""
    pixels = np.random.default_rng(42).integers(0, 255, size=(height, width, 1), dtype=np.uint8)
    return tf.io.encode_png(tf.convert_to_tensor(pixels))


@pytest.mark.parametrize(
    ("width", "height"),
    [
        (400, 100),  # narrow: the width limit never binds
        (2000, 100),  # wide: at HEIGHT it would exceed the limit
        (6000, 100),  # very wide: far over the limit
        (100, 400),  # taller than it is wide
        (50, 50),  # tiny
    ],
)
def test_the_height_is_always_exactly_what_was_asked_for(width: int, height: int) -> None:
    """The encoder's input layer declares this height; nothing may vary it.

    This is the regression that broke training and evaluation on two thirds of
    the dolores dataset: with `preserve_aspect_ratio` and a width limit, the
    width bound first and the height came out short.
    """
    result = preprocess_image(
        an_image(width, height), height=HEIGHT, transformations=[], max_image_width=1500
    )

    assert result.shape[0] == HEIGHT


def test_a_wide_image_is_capped_at_the_maximum_width() -> None:
    result = preprocess_image(
        an_image(6000, 100), height=HEIGHT, transformations=[], max_image_width=1500
    )

    assert tuple(result.shape) == (HEIGHT, 1500, 1)


def test_a_narrow_image_keeps_its_aspect_ratio() -> None:
    """Below the cap nothing is squashed: the width follows from the height."""
    result = preprocess_image(
        an_image(400, 100), height=HEIGHT, transformations=[], max_image_width=1500
    )

    assert tuple(result.shape) == (HEIGHT, round(400 * HEIGHT / 100), 1)


def test_without_a_cap_the_width_is_whatever_the_aspect_ratio_gives() -> None:
    result = preprocess_image(
        an_image(6000, 100), height=HEIGHT, transformations=[], max_image_width=None
    )

    assert tuple(result.shape) == (HEIGHT, round(6000 * HEIGHT / 100), 1)


def test_the_result_is_a_single_channel_float_in_the_unit_range() -> None:
    result = preprocess_image(
        an_image(400, 100), height=HEIGHT, transformations=[], max_image_width=1500
    )

    assert result.dtype == tf.float32
    assert result.shape[2] == 1
    assert float(tf.reduce_min(result)) >= 0.0 and float(tf.reduce_max(result)) <= 1.0


def midrange_pixel_count(image: tf.Tensor) -> int:
    """How many pixels are neither black nor white."""
    return int(tf.reduce_sum(tf.cast((image > 0.01) & (image < 0.99), tf.int32)))


def test_thresholding_binarizes() -> None:
    """An image that needs no rescaling comes out purely black and white."""
    already_the_right_size = an_image(384, HEIGHT)

    result = preprocess_image(
        already_the_right_size,
        height=HEIGHT,
        transformations=["threshold:0.5:0.5"],
        max_image_width=1500,
    )

    assert midrange_pixel_count(result) == 0


def test_thresholding_happens_before_the_rescale() -> None:
    """Worth pinning down, because it means the output is not always binary.

    Transformations run at the source resolution and the antialiased resize
    follows, so downscaling a thresholded image interpolates grays back into
    it. That is the useful order — the threshold is chosen against the scan's
    own contrast, and antialiased strokes suit a CNN better than jagged ones —
    but it does mean `threshold` is not a promise that the tensor is binary.
    """
    downscaled = preprocess_image(
        an_image(4000, 1000),
        height=HEIGHT,
        transformations=["threshold:0.5:0.5"],
        max_image_width=1500,
    )

    assert midrange_pixel_count(downscaled) > 0


def test_an_unknown_transformation_is_refused() -> None:
    with pytest.raises(ValueError, match="is unknown"):
        preprocess_image(
            an_image(400, 100),
            height=HEIGHT,
            transformations=["sharpen:2"],
            max_image_width=1500,
        )
