"""Rendering a model's predictions beside the gold data they should match.

Samples are ordered worst-error-last, so scrolling the page walks from what the
model reads perfectly to what defeats it, with the median error in the middle.
That ordering is the whole value of the page: a single average error rate says
a model is imperfect, and this says how.
"""

import html
import random
from pathlib import Path

from ..data.zeus_dataset import ZeusDataset
from ..evaluation.metrics import SER


def visualize_predictions(
    title: str,
    dataset: ZeusDataset,
    predictions_lmx: list[str],
    output_html_path: Path,
    sample_count: int = 100,
) -> None:
    """Write an HTML page pairing each prediction with its image and gold LMX.

    No model is involved: the predictions have already been made, and this
    reads the images out of the dataset they were made from. Which is why the
    dataset must be the one that was evaluated — the nth prediction is matched
    to the nth sample by position, and nothing else could detect a mismatch.

    :param title: Shown in the heading, to tell one page from another.
    :param dataset: The dataset the predictions were made on.
    :param predictions_lmx: One predicted LMX string per sample, in order.
    :param output_html_path: The `.html` file to write. Images go in a folder
        beside it.
    :param sample_count: Show at most this many samples, chosen at random with
        a fixed seed so that the page is the same page every time.
    """
    assert output_html_path.suffix == ".html"
    assert len(dataset.samples) == len(predictions_lmx), (
        "Given dataset has different number of samples than "
        + "the precitions LMX file. Did you provide the correct dataset?"
    )

    # Permuted before truncating, so a subsample is spread over the dataset
    # rather than being its first hundred samples. Seeded, so that reloading
    # the page after a change compares like with like.
    sample_indices = list(range(len(predictions_lmx)))
    random.Random(42).shuffle(sample_indices)
    if len(sample_indices) > sample_count:
        sample_indices = sample_indices[:sample_count]

    output_html_path.parent.mkdir(parents=True, exist_ok=True)
    images_folder_path = output_html_path.parent / (output_html_path.stem + "-imgs")
    images_folder_path.mkdir(exist_ok=True)

    # (sample_index, image, gold, predicted, ser)
    items: list[tuple[int, bytes, str, str, float]] = []
    for sample_index in sample_indices:
        sample = dataset.samples[sample_index]
        gold_lmx: str = sample.lmx
        pred_lmx: str = predictions_lmx[sample_index]
        ser: float = SER.compute([gold_lmx], [pred_lmx])
        items.append((sample_index, sample.image, gold_lmx, pred_lmx, ser))

    items.sort(key=lambda item: item[4])

    document = f"<html><body><h1>{html.escape(title)} @ {html.escape(dataset.name)}</h1>"
    for sample_index, image, gold_lmx, pred_lmx, ser in items:
        (images_folder_path / f"{sample_index}.jpg").write_bytes(image)

        document += "<div>"
        document += f'<img src="{images_folder_path.name}/{sample_index}.jpg">'
        document += f"<p>SER: <strong>{ser:.2f}</strong></p>"
        # Escaped because `<unk>` is a real token and survives decoding: written
        # raw, a browser reads it as an unknown tag and shows nothing, hiding
        # exactly the token worth noticing in a prediction.
        document += f"<p>Gold LMX: <code>{html.escape(gold_lmx)}</code></p>"
        document += f"<p>Predicted LMX: <code>{html.escape(pred_lmx)}</code></p>"
        document += "</div>"

    document += "</body></html>"
    output_html_path.write_text(document)

    print("Visualisation has been written to", output_html_path)
