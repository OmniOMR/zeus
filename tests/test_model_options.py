"""Tests for what a snapshot says it reads.

This is a wire contract, not a note to the reader: it becomes the input half of
a Musibot worker's signature and so decides which images the model is sent. A
grandstaff model told to read staves does not fail — it transcribes confidently
and wrongly — so the value has to be right, and these tests pin down that it
travels with the snapshot and cannot hold nonsense.
"""

from pathlib import Path

import pytest

from zeus.model.model_options import DEFAULT_INPUT_SUBDIVISIONS, ModelOptions


def test_a_model_reads_grandstaves_unless_it_says_otherwise() -> None:
    assert ModelOptions().input_subdivisions == DEFAULT_INPUT_SUBDIVISIONS


def test_it_behaves_as_a_set() -> None:
    """Written as a list, but order and repetition carry no meaning."""
    one_way = ModelOptions(input_subdivisions=["Staves", "Grandstaves"])
    the_other = ModelOptions(input_subdivisions=["Grandstaves", "Staves", "Grandstaves"])

    assert one_way == the_other
    assert one_way.input_subdivisions == ["Grandstaves", "Staves"]


def test_accepts_answers_what_the_model_can_read() -> None:
    options = ModelOptions(input_subdivisions=["Staves"])

    assert options.accepts("Staves")
    assert not options.accepts("Grandstaves")
    assert not options.accepts("Systems")


def test_a_model_reading_both_accepts_both() -> None:
    options = ModelOptions(input_subdivisions=["Staves", "Grandstaves"])

    assert options.accepts("Staves")
    assert options.accepts("Grandstaves")


def test_an_unknown_subdivision_is_refused() -> None:
    """Only the three the Musicorpus Specification defines exist."""
    with pytest.raises(ValueError, match="Unknown page subdivision"):
        ModelOptions(input_subdivisions=["Measures"])


def test_reading_nothing_is_refused() -> None:
    with pytest.raises(ValueError, match="at least one subdivision"):
        ModelOptions(input_subdivisions=[])


def test_options_survive_a_trip_through_a_snapshot(tmp_path: Path) -> None:
    written = ModelOptions(input_subdivisions=["Systems", "Staves"])

    written.write_to_model_folder(tmp_path)
    loaded = ModelOptions.from_model_folder(tmp_path)

    assert loaded == written
    assert (tmp_path / "model_options.yaml").is_file()


def test_a_snapshot_without_the_file_reads_as_a_grandstaff_model(tmp_path: Path) -> None:
    """Every snapshot written before this file existed, which must keep working.

    Such a snapshot can also be corrected by hand — writing the file into its
    folder is enough, and nothing has to be retrained.
    """
    assert not (tmp_path / "model_options.yaml").exists()

    assert ModelOptions.from_model_folder(tmp_path).input_subdivisions == ["Grandstaves"]


def test_the_stored_file_is_readable_yaml(tmp_path: Path) -> None:
    """It is meant to be edited by hand on an existing snapshot."""
    ModelOptions(input_subdivisions=["Staves"]).write_to_model_folder(tmp_path)

    content = (tmp_path / "model_options.yaml").read_text()

    assert "input_subdivisions" in content
    assert "Staves" in content
