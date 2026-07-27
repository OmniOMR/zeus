from dataclasses import dataclass
from typing import Literal
from pathlib import Path


@dataclass
class MusicorpusSample:
    """
    Represents a Zeus sample that is to be extracted from a MusiCorpus dataset.
    Contains just identification, but no actual data.
    """

    musicorpus_path: Path
    """Path to the folder that contains the sample
    (transcription.musicxml and image.jpg). I.e. path to the
    staff or grandstaff folder."""

    subdivision: Literal["Staves", "Grandstaves"]
    """Is this sample a staff or a grandstaff?
    Corresponds to the subdivision folder name."""

    page_name: str
    """Name of the page in the MusiCorpus dataset
    from which this sample is taken"""

    def get_zeus_sample_name(self) -> str:
        """Computes the name of the zeus dataset sample"""
        return f"samples/{self.page_name}/{self.subdivision}_{self.musicorpus_path.name}"
