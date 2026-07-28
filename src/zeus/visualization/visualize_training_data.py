"""Rendering training data as the model actually receives it.

The point is to see what augmentation and preprocessing did — not what is on
disk, but what comes out of the input pipeline one step before the encoder. So
the images written here are the model's own input tensors, encoded back to PNG.
"""

import html
from pathlib import Path

import tensorflow as tf

from ..data.shuffled_view import ShuffledView
from ..model.architecture_options import ArchitectureOptions
from ..model.construct_tf_dataset import construct_tf_dataset
from ..model.token_map import TokenMap
from ..model.training_options import TrainingOptions


def visualize_training_data(
    shuffled_train_dataset: ShuffledView,
    architecture_options: ArchitectureOptions,
    token_map: TokenMap,
    training_options: TrainingOptions,
    output_folder_path: Path,
    sample_count: int = 100,
) -> None:
    """Write images and an index.html showing what training will feed the model.

    No trained model is involved, and none is needed: what the input pipeline
    produces depends on the architecture's image height, the token map and the
    training options, not on any weights.

    :param shuffled_train_dataset: The training data, in the order training
        would see it.
    :param architecture_options: Used for the image height to normalize to.
    :param token_map: Used to encode LMX and to decode it back for display, so
        that what is shown is what the model is actually told.
    :param training_options: Batch size, augmentations, seed — the settings
        whose effect this is meant to reveal.
    :param output_folder_path: Written to, and created if missing.
    :param sample_count: Stop after this many samples.
    """
    train_tf_dataset = construct_tf_dataset(
        shuffled_view=shuffled_train_dataset,
        architecture_options=architecture_options,
        token_map=token_map,
        training_or_inference_options=training_options,
    )

    print("There are", len(train_tf_dataset), "batches in the dataset")

    output_folder_path.mkdir(parents=True, exist_ok=True)
    images_folder_path = output_folder_path / "images"
    images_folder_path.mkdir(exist_ok=True)
    html_file_path = output_folder_path / "index.html"

    document = f"<html><body><h1>{html.escape(shuffled_train_dataset.dataset.name)}</h1>"
    sample_index = 0

    for batch_index, (batch_images, batch_annotations) in enumerate(train_tf_dataset):
        assert batch_images.shape[0] == batch_annotations.shape[0]
        document += f"<h2>Batch {batch_index}</h2>"

        for i in range(batch_images.shape[0]):
            png_bytes = tf.image.encode_png(
                tf.cast(batch_images[i].to_tensor() * 255, tf.uint8)
            ).numpy()
            (images_folder_path / f"{sample_index}.png").write_bytes(png_bytes)

            lmx = token_map.indices_to_lmx(batch_annotations[i].numpy())

            document += "<div>"
            document += f'<img src="images/{sample_index}.png">'
            # Escaped because `<unk>` is a real token and survives decoding:
            # written raw, a browser reads it as an unknown tag and shows
            # nothing, hiding exactly the token worth noticing.
            document += f"<p>LMX: <code>{html.escape(lmx)}</code></p>"
            document += "</div>"

            sample_index += 1
            if sample_index >= sample_count:
                break

        if sample_index >= sample_count:
            break

    document += "</body></html>"
    html_file_path.write_text(document)

    print("Visualisation has been written to", output_folder_path)
