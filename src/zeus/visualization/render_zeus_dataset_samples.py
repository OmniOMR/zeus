from pathlib import Path
from ..data.SamplesFile import SamplesFile
import tqdm
from lmx.musescore.MuseScore import MuseScore
from lmx.musescore.render_staff import render_staff
from lmx.musicxml.io.read_musicxml_tree_from_file \
    import read_musicxml_tree_from_file
import xml.etree.ElementTree as ET


def render_zeus_dataset_samples(
        ms: MuseScore,
        samples_path: Path,
        image_suffix: str,
        render_invisible: bool,
        batch_size: int
):
    samples = SamplesFile.load(samples_path)

    # group into batches
    sample_batches = [
        samples[i:i+batch_size]
        for i in range(0, len(samples), batch_size)
    ]
    
    for batch in tqdm.tqdm(sample_batches):
        part_elements = [
            _load_part_from_file(sample.path.with_suffix(".musicxml"))
            for sample in batch
        ]
        output_png_files = [
            sample.path
                .with_name(sample.path.name + image_suffix)
                .with_suffix(".png")
            for sample in batch
        ]
        render_staff(
            ms=ms,
            part_element=part_elements,
            output_png_file=output_png_files,
            render_invisible_attributes=render_invisible,
            page_width_tenths=6_000,
        )


def _load_part_from_file(file_path: Path) -> ET.Element:
    musicxml_tree = read_musicxml_tree_from_file(file_path)
    parts = musicxml_tree.findall("part")
    assert len(parts) == 1, \
        f"The sample {file_path} does not have a single <part> element."
    return parts[0]
