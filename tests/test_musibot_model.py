"""Tests for the Zeus half of the worker IPC: what it announces and what it writes.

A stub stands in for the model, so no weights are loaded and no TensorFlow is
imported. What is under test here is the file handling and the reporting, which
is where a model integration actually goes wrong.
"""

import io
import json
from pathlib import Path
from typing import Any

import pytest

from zeus.model.inference_options import InferenceOptions
from zeus.model.model_options import ModelOptions
from zeus.musibot.protocol import Execution, Reporter
from zeus.musibot.zeus_model import ZeusMusibotModel

A_MEASURE = "measure key:fifths:0 time beats:4 beat-type:4 clef:G2 C4 quarter"


class StubZeus:
    """Predicts a fixed answer, or whatever the test asked for."""

    def __init__(self, model_options: ModelOptions, predictions: list[str] | None = None) -> None:
        self.model_options = model_options
        self.predictions = predictions
        self.seen_batches: list[int] = []

    def predict(
        self,
        images: list[bytes],
        inference_options: InferenceOptions,
        with_progress_bar: bool = False,
    ) -> list[str]:
        self.seen_batches.append(len(images))
        if self.predictions is not None:
            return self.predictions[: len(images)]
        return [A_MEASURE] * len(images)


def a_page(pages_dir: Path, page: str = "PAGE", subdivision: str = "Staves", count: int = 1):
    """Lay out a page folder the way a worker head stages one."""
    for index in range(1, count + 1):
        folder = pages_dir / page / subdivision / str(index)
        folder.mkdir(parents=True)
        (folder / "image.jpg").write_bytes(b"pretend this is a JPEG")


def a_model(
    pages_dir: Path,
    subdivisions: list[str] | None = None,
    predictions: list[str] | None = None,
    write_lmx: bool = True,
    name: str | None = "zeus-solo",
    version: str | None = "2026-07-28-143052-e50",
) -> ZeusMusibotModel:
    options = ModelOptions(
        input_subdivisions=subdivisions or ["Staves"],
        musibot_model_name=name,
        musibot_model_version=version,
    )
    return ZeusMusibotModel(
        zeus=StubZeus(options, predictions),
        snapshot_path=Path("models/solo26.model"),
        pages_dir=pages_dir,
        inference_options=InferenceOptions(batch_size=4),
        write_lmx=write_lmx,
    )


def an_execution(execution_id: str = "e1", input_path: str = "Staves/1/image.jpg") -> Execution:
    return Execution(execution_id=execution_id, page="PAGE", input=[input_path])


def reports_from(results: io.StringIO) -> dict[str, dict[str, Any]]:
    """The last report for each execution."""
    reports = {}
    for line in results.getvalue().splitlines():
        message = json.loads(line)
        if message["type"] in ("completed", "failed"):
            reports[message["execution_id"]] = message
    return reports


# --- what it announces ------------------------------------------------------


def test_it_announces_the_identity_from_the_snapshot(tmp_path: Path) -> None:
    description = a_model(tmp_path).describe()

    assert description["name"] == "zeus-solo"
    assert description["version"] == "2026-07-28-143052-e50"


def test_an_unnamed_snapshot_falls_back_to_its_folder_name(tmp_path: Path) -> None:
    description = a_model(tmp_path, name=None, version=None).describe()

    assert description["name"] == "zeus"
    assert description["version"] == "solo26"


def test_the_signature_follows_what_the_model_reads(tmp_path: Path) -> None:
    description = a_model(tmp_path, subdivisions=["Staves"]).describe()

    assert description["signature"]["input"] == ["Staves/1/image.jpg"]
    assert description["signature"]["output"] == [
        "Staves/1/transcription.musicxml",
        "Staves/1/transcription.lmx",
    ]


def test_a_model_reading_both_announces_both(tmp_path: Path) -> None:
    description = a_model(tmp_path, subdivisions=["Staves", "Grandstaves"]).describe()

    assert description["signature"]["input"] == [
        "Grandstaves/1/image.jpg",
        "Staves/1/image.jpg",
    ]


def test_batching_is_advertised(tmp_path: Path) -> None:
    """It is what lets a batch fill one forward pass."""
    assert a_model(tmp_path).describe()["supports_batching"] is True


def test_lmx_is_left_out_of_the_signature_when_it_is_not_written(tmp_path: Path) -> None:
    description = a_model(tmp_path, write_lmx=False).describe()

    assert description["signature"]["output"] == ["Staves/1/transcription.musicxml"]


# --- what it writes ---------------------------------------------------------


def test_it_writes_the_transcription_beside_the_image(tmp_path: Path) -> None:
    a_page(tmp_path)
    results = io.StringIO()

    a_model(tmp_path).handle([an_execution()], Reporter(results))

    staff = tmp_path / "PAGE" / "Staves" / "1"
    assert (staff / "transcription.musicxml").is_file()
    assert (staff / "transcription.lmx").read_text().strip() == A_MEASURE
    assert reports_from(results)["e1"]["type"] == "completed"


def test_the_musicxml_is_a_score(tmp_path: Path) -> None:
    a_page(tmp_path)

    a_model(tmp_path).handle([an_execution()], Reporter(io.StringIO()))

    musicxml = (tmp_path / "PAGE" / "Staves" / "1" / "transcription.musicxml").read_text()
    assert "<score-partwise" in musicxml


def test_no_lmx_file_when_it_was_turned_off(tmp_path: Path) -> None:
    a_page(tmp_path)

    a_model(tmp_path, write_lmx=False).handle([an_execution()], Reporter(io.StringIO()))

    staff = tmp_path / "PAGE" / "Staves" / "1"
    assert (staff / "transcription.musicxml").is_file()
    assert not (staff / "transcription.lmx").exists()


def test_a_whole_batch_goes_through_one_forward_pass(tmp_path: Path) -> None:
    a_page(tmp_path, count=3)
    model = a_model(tmp_path)

    model.handle(
        [an_execution(f"e{i}", f"Staves/{i}/image.jpg") for i in (1, 2, 3)],
        Reporter(io.StringIO()),
    )

    stub = model.zeus
    assert isinstance(stub, StubZeus)
    assert stub.seen_batches == [3]


# --- what it refuses --------------------------------------------------------


def test_a_missing_image_fails_only_its_own_execution(tmp_path: Path) -> None:
    a_page(tmp_path, count=1)
    results = io.StringIO()

    a_model(tmp_path).handle(
        [an_execution("e1"), an_execution("e2", "Staves/9/image.jpg")],
        Reporter(results),
    )

    reports = reports_from(results)
    assert reports["e1"]["type"] == "completed"
    assert reports["e2"]["type"] == "failed"
    assert "no file" in reports["e2"]["error"].lower()


def test_a_subdivision_the_model_cannot_read_is_refused(tmp_path: Path) -> None:
    """The check that stops a grandstaff model transcribing staves."""
    a_page(tmp_path, subdivision="Grandstaves")
    results = io.StringIO()

    a_model(tmp_path, subdivisions=["Staves"]).handle(
        [an_execution("e1", "Grandstaves/1/image.jpg")], Reporter(results)
    )

    report = reports_from(results)["e1"]
    assert report["type"] == "failed"
    assert "reads Staves" in report["error"]


def test_a_path_escaping_the_page_is_refused(tmp_path: Path) -> None:
    a_page(tmp_path)
    (tmp_path / "secrets.jpg").write_bytes(b"not yours")
    results = io.StringIO()

    a_model(tmp_path).handle([an_execution("e1", "../../secrets.jpg")], Reporter(results))

    report = reports_from(results)["e1"]
    assert report["type"] == "failed"
    assert "escapes its page folder" in report["error"]


def test_more_than_one_input_file_is_refused(tmp_path: Path) -> None:
    a_page(tmp_path, count=2)
    results = io.StringIO()

    execution = Execution(
        execution_id="e1", page="PAGE", input=["Staves/1/image.jpg", "Staves/2/image.jpg"]
    )
    a_model(tmp_path).handle([execution], Reporter(results))

    assert "exactly one image" in reports_from(results)["e1"]["error"]


def test_an_unreadable_prediction_fails_only_its_own_execution(tmp_path: Path) -> None:
    """A model's output is a prediction, not a promise."""
    a_page(tmp_path, count=2)
    results = io.StringIO()

    a_model(tmp_path, predictions=["A2 E2 A2", A_MEASURE]).handle(
        [an_execution("e1", "Staves/1/image.jpg"), an_execution("e2", "Staves/2/image.jpg")],
        Reporter(results),
    )

    reports = reports_from(results)
    assert reports["e1"]["type"] == "failed"
    assert "Could not decode" in reports["e1"]["error"]
    assert reports["e2"]["type"] == "completed"


def test_nothing_is_predicted_when_every_execution_is_invalid(tmp_path: Path) -> None:
    a_page(tmp_path)
    model = a_model(tmp_path)

    model.handle([an_execution("e1", "Staves/9/image.jpg")], Reporter(io.StringIO()))

    stub = model.zeus
    assert isinstance(stub, StubZeus)
    assert stub.seen_batches == []


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        ("Staves/1/image.jpg", "Staves"),
        ("Grandstaves/1-2/image.jpg", "Grandstaves"),
        ("image.jpg", None),
    ],
)
def test_the_subdivision_is_read_off_the_path(path: str, expected: str | None) -> None:
    assert ZeusMusibotModel._subdivision_of(path) == expected
