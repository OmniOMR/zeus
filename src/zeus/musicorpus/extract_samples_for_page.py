from pathlib import Path
from typing import Iterable
from .MusicorpusSample import MusicorpusSample


def extract_samples_for_page(
        page_path: Path,
        take_staves: bool,
        take_grandstaves: bool,
) -> Iterable[MusicorpusSample]:
    """
    Given a MusiCorpus page, it extracts samples defined within it.
    
    Since folders may exist and be empty, we are interested in samples
    where MusicXML files are actually present. The presence of the image
    file will be checked later since it depends on whether re-cropping occurs.

    :param page_path: Path to the page folder in the MusiCorpus datset.
    :param take_staves: Emit staves as samples.
    :param take_grandstaves: Emit grandstaves as samples.
    """

    if not page_path.is_dir():
        raise Exception(f"There is no folder at path {page_path}")

    # iterate staves
    if take_staves:
        for staff_path in (page_path / "Staves").iterdir():
            if (staff_path / "transcription.musicxml").exists():
                yield MusicorpusSample(
                    musicorpus_path=staff_path,
                    subdivision="Staves",
                    page_name=page_path.name,
                )

    # iterate grandstaves
    if take_grandstaves:
        for grandstaff_path in (page_path / "Grandstaves").iterdir():
            if (grandstaff_path / "transcription.musicxml").exists():
                yield MusicorpusSample(
                    musicorpus_path=grandstaff_path,
                    subdivision="Grandstaves",
                    page_name=page_path.name,
                )
