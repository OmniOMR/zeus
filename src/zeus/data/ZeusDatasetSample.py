from dataclasses import dataclass
from pathlib import Path


@dataclass
class ZeusDatasetSample:
    """
    One sample of a Zeus dataset, raw-unparsed, loaded in-memory.
    A list of these is pickled to speed up model training.
    """
    
    sample_name: str
    """
    Name of the sample, taken exactly from the samples file.

    Also acts as a posix path relative to the samples file
    to the base name of the JPG and LMX files of the dataset.

    Example sample name:
    `samples/chopin/mazurkas/mazurka17-2/maj2_down_m-0-3`
    """

    image: bytes
    """
    The binary content of the image file containing the music notation.
    May be PNG of JPG. Can be loaded by OpenCV imread function.
    """

    lmx: str
    """
    LMX string representation of the music notation.
    Single-line, tokens separated by spaces. No newlines.
    """

    musicxml: str | None
    """
    XML string loaded from the .musicxml file.
    May be None as it's only needed for MusicXML-based evaluation.
    """

    def __postinit__(self):
        assert Path(self.sample_name).as_posix() == self.sample_name
        assert len(self.image) > 0
        assert "\n" not in self.lmx
        assert "\r" not in self.lmx
