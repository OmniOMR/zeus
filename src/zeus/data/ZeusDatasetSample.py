from pathlib import Path
from dataclasses import dataclass


@dataclass
class ZeusDatasetSample:
    """One sample from a samples file"""
    name: str
    """The exact string that is in the samples file on one line."""
    
    path: Path
    """
    Absolute path to the sample, without sample type suffix.

    The ".jpg" or "_distorted.jpg" or ".lmx" must be added
    to actually point to a file.
    """
