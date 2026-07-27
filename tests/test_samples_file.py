"""Tests for the samples file — the index of a Zeus dataset split.

A samples file holds sample *names*, and every name doubles as a path relative
to the file itself. That second role is the part worth pinning down: it is what
lets a dataset folder be moved without rewriting its index.
"""

from pathlib import Path

from zeus.data.SamplesFile import SamplesFile


def test_a_samples_file_survives_a_trip_through_the_disk(tmp_path: Path) -> None:
    written = SamplesFile.empty(tmp_path / "samples.train.txt")
    written.append("samples/chopin/mazurka17-2/maj2_down_m-0-3")
    written.append("samples/chopin/mazurka17-2/maj2_down_m-4-7")
    written.write()

    loaded = SamplesFile.load(tmp_path / "samples.train.txt")

    assert [sample.name for sample in loaded] == [sample.name for sample in written]
    assert len(loaded) == 2


def test_a_sample_name_resolves_against_the_samples_file(tmp_path: Path) -> None:
    """The name is relative to the file's folder, not to the current directory."""
    samples = SamplesFile.empty(tmp_path / "dolores" / "samples.all.txt")

    sample = samples.append("samples/page-1/staff-2")

    assert sample.path == tmp_path / "dolores" / "samples" / "page-1" / "staff-2"


def test_trailing_whitespace_is_stripped_on_load(tmp_path: Path) -> None:
    """Names are used as path fragments, so a stray `\\r` would be part of one."""
    path = tmp_path / "samples.test.txt"
    path.write_text("samples/first\r\nsamples/second\n")

    samples = SamplesFile.load(path)

    assert [sample.name for sample in samples] == ["samples/first", "samples/second"]


def test_a_slice_is_a_samples_file_of_its_own(tmp_path: Path) -> None:
    samples = SamplesFile.empty(tmp_path / "samples.all.txt")
    for index in range(5):
        samples.append(f"samples/{index}")

    subset = samples[1:3]

    assert isinstance(subset, SamplesFile)
    assert [sample.name for sample in subset] == ["samples/1", "samples/2"]
    # Renamed, so that writing a slice cannot overwrite the file it came from.
    assert subset.file_path != samples.file_path


def test_indexing_yields_one_sample(tmp_path: Path) -> None:
    samples = SamplesFile.empty(tmp_path / "samples.all.txt")
    samples.append("samples/only")

    assert samples[0].name == "samples/only"
