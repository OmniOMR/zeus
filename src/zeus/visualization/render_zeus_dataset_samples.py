import xml.etree.ElementTree as ET
from pathlib import Path

import tqdm
from lmx.musescore.MuseScore import MuseScore
from lmx.musescore.render_staff import render_staff
from lmx.musicxml.io.read_musicxml_tree_from_file import read_musicxml_tree_from_file

from ..data.samples_file import Sample, SamplesFile


def render_zeus_dataset_samples(
    ms: MuseScore,
    samples_path: Path,
    image_suffix: str,
    render_invisible: bool,
    batch_size: int,
    page_width_tenths: int,
):
    samples = SamplesFile.load(samples_path)

    # samples that were not rendered
    failed_samples: list[Sample] = []

    # group into batches
    sample_batches = [samples[i : i + batch_size] for i in range(0, len(samples), batch_size)]

    for batch in tqdm.tqdm(sample_batches):
        part_elements = [
            _load_part_from_file(sample.path.with_suffix(".musicxml")) for sample in batch
        ]
        output_png_files = [
            sample.path.with_name(sample.path.name + image_suffix).with_suffix(".png")
            for sample in batch
        ]
        overflown_sample_indices: list[int] = []
        render_staff(
            ms=ms,
            part_element=part_elements,
            output_png_file=output_png_files,
            render_invisible_attributes=render_invisible,
            page_width_tenths=page_width_tenths,
            on_page_overflow="return-index",
            overflown_sample_indices=overflown_sample_indices,
        )
        for i in overflown_sample_indices:
            failed_samples.append(batch[i])

    if len(failed_samples) > 0:
        print()
        print("Not all samples were rendered, due to page overflow issues.")
        print("These are the failed samples:")
        print("---------------------------------")
        for sample in failed_samples:
            print(sample.name)


def _load_part_from_file(file_path: Path) -> ET.Element:
    musicxml_tree = read_musicxml_tree_from_file(file_path)
    parts = musicxml_tree.findall("part")
    assert len(parts) == 1, f"The sample {file_path} does not have a single <part> element."
    return parts[0]
