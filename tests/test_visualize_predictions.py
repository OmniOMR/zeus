"""Tests for the predictions visualization.

It needs no model and no TensorFlow — the predictions have already been made —
so these tests run on a handful of fabricated samples in milliseconds.
"""

import re
from pathlib import Path

import pytest

from zeus.data.zeus_dataset import ZeusDataset, ZeusDatasetSample
from zeus.visualization.visualize_predictions import visualize_predictions

AN_IMAGE = b"not really a JPEG, and nothing here decodes it"


def a_dataset(lmx_strings: list[str]) -> ZeusDataset:
    return ZeusDataset(
        name="test-dataset",
        samples=[
            ZeusDatasetSample(sample_name=f"s{i}", image=AN_IMAGE, lmx=lmx, musicxml=None)
            for i, lmx in enumerate(lmx_strings)
        ],
    )


def test_it_writes_a_page_and_the_images_beside_it(tmp_path: Path) -> None:
    dataset = a_dataset(["measure C4 quarter", "measure D4 half"])

    visualize_predictions(
        title="run-1",
        dataset=dataset,
        predictions_lmx=["measure C4 quarter", "measure D4 half"],
        output_html_path=tmp_path / "predictions.html",
    )

    assert (tmp_path / "predictions.html").is_file()
    assert len(list((tmp_path / "predictions-imgs").iterdir())) == 2


def test_samples_are_ordered_worst_last(tmp_path: Path) -> None:
    """The ordering is the whole value of the page: it says *how* a model fails."""
    dataset = a_dataset(["measure C4 quarter"] * 4)

    visualize_predictions(
        title="run-1",
        dataset=dataset,
        predictions_lmx=[
            "measure C4 quarter",  # perfect
            "measure C4 half",  # one token wrong
            "measure",  # two dropped
            "",  # nothing at all
        ],
        output_html_path=tmp_path / "predictions.html",
    )

    document = (tmp_path / "predictions.html").read_text()
    reported = [float(x) for x in re.findall(r"SER: <strong>([\d.]+)</strong>", document)]

    assert reported == sorted(reported)
    assert reported[0] == 0.0


def test_an_unknown_token_is_escaped_rather_than_swallowed(tmp_path: Path) -> None:
    """`<unk>` is a real token that survives decoding into the LMX.

    Written raw into the page a browser reads it as an unknown tag and displays
    nothing — silently hiding the one token in a prediction most worth seeing.
    """
    dataset = a_dataset(["measure C4 quarter"])

    visualize_predictions(
        title="run-1",
        dataset=dataset,
        predictions_lmx=["measure <unk> quarter"],
        output_html_path=tmp_path / "predictions.html",
    )

    document = (tmp_path / "predictions.html").read_text()
    assert "&lt;unk&gt;" in document
    assert "<unk>" not in document


def test_the_title_is_escaped_too(tmp_path: Path) -> None:
    visualize_predictions(
        title="run <b>one</b>",
        dataset=a_dataset(["measure C4 quarter"]),
        predictions_lmx=["measure C4 quarter"],
        output_html_path=tmp_path / "predictions.html",
    )

    assert "<b>one</b>" not in (tmp_path / "predictions.html").read_text()


def test_a_subsample_is_capped_but_the_page_is_still_written(tmp_path: Path) -> None:
    dataset = a_dataset(["measure C4 quarter"] * 20)

    visualize_predictions(
        title="run-1",
        dataset=dataset,
        predictions_lmx=["measure C4 quarter"] * 20,
        output_html_path=tmp_path / "predictions.html",
        sample_count=5,
    )

    assert (tmp_path / "predictions.html").read_text().count("<img") == 5


def test_the_wrong_dataset_is_refused(tmp_path: Path) -> None:
    """Predictions are matched to samples by position and nothing else, so a
    length mismatch is the only detectable sign of the wrong dataset."""
    with pytest.raises(AssertionError, match="different number of samples"):
        visualize_predictions(
            title="run-1",
            dataset=a_dataset(["measure C4 quarter"] * 3),
            predictions_lmx=["measure C4 quarter"],
            output_html_path=tmp_path / "predictions.html",
        )


def test_the_output_must_be_an_html_file(tmp_path: Path) -> None:
    with pytest.raises(AssertionError):
        visualize_predictions(
            title="run-1",
            dataset=a_dataset(["measure C4 quarter"]),
            predictions_lmx=["measure C4 quarter"],
            output_html_path=tmp_path / "predictions.txt",
        )
