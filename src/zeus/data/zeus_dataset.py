import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tqdm

from .samples_file import SamplesFile


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


class ZeusDataset:
    """
    Holds the contents of a Zeus dataset, loaded up in memory.
    May have been loaded either from a pickle file or from
    the unpickled form.
    """

    def __init__(self, name: str, samples: list[ZeusDatasetSample]) -> None:
        self.name = name
        """Human-readable name of the dataset, used in logs, must be path-safe"""

        self.samples = samples
        """Inidividual samples of the dataset, ordered in the same
        way as the samples txt file"""

    @staticmethod
    def load_from_samples_file(
        samples_file_path: Path,
        image_suffix: str,
        with_musicxml: bool,
        show_progress_bar: bool = False,
        benevolent: bool = False,
    ) -> "ZeusDataset":
        """
        Loads a Zeus dataset from its folder-representation,
        specifically loads only one slice from the given samples file.
        This loading takes a long time, so a progress bar may be shown
        and for training, the pickled representation should be used instead.

        :param samples_file_path: Path to the samples.split.txt file.
        :param image_suffix: Paths to images may be suffixed to load
            e.g. camera grandstaff LMX dataset.
        :param with_musicxml: Whether to load MusicXML files as well or not.
        :param show_progress_bar: Whether to show a tqdm progress bar while loading.
        :param benevolent: Skip samples with missing images without raising.
        """
        zeus_dataset_samples: list[ZeusDatasetSample] = []

        samples = SamplesFile.load(samples_file_path)
        with tqdm.tqdm(total=len(samples), disable=not show_progress_bar) as pbar:
            for sample in samples:
                # load image
                image: bytes | None = None
                for extension in [".jpg", ".png"]:
                    image_path = sample.path.with_name(sample.path.name + image_suffix).with_suffix(
                        extension
                    )
                    if image_path.exists():
                        image = image_path.read_bytes()
                        break
                if image is None:
                    if benevolent:
                        pbar.update(1)
                        continue
                    raise RuntimeError(
                        f"Sample is missing an image file: {sample.name}\n"
                        + "Set the 'benevolent' flag if such samples should be ignored."
                    )

                # load lmx
                lmx = sample.path.with_suffix(".lmx").read_text(encoding="utf-8").rstrip("\r\n")

                # load musicxml
                musicxml: str | None = None
                if with_musicxml:
                    musicxml = sample.path.with_suffix(".musicxml").read_text(encoding="utf-8")

                zeus_dataset_samples.append(
                    ZeusDatasetSample(
                        sample_name=sample.name,
                        image=image,
                        lmx=lmx,
                        musicxml=musicxml,
                    )
                )

                pbar.update(1)

        return ZeusDataset(
            name=samples_file_path.as_posix(),
            samples=zeus_dataset_samples,
        )

    @staticmethod
    def load_from_pickle_file(pickle_path: Path) -> "ZeusDataset":
        """Loads a dataset from its pickled representation"""
        with open(str(pickle_path), "rb") as file:
            samples = pickle.load(file)
            assert type(samples) is list
            assert len(samples) > 0
            assert type(samples[0]) is ZeusDatasetSample

        name = pickle_path.as_posix()
        if name.startswith("datasets/"):
            name = name[len("datasets/") :]
        name = name.replace("/", "_")

        return ZeusDataset(
            name=name,
            samples=samples,
        )

    def write_to_pickle_file(self, pickle_path: Path):
        """Writes the dataset to a pickle file"""
        with open(str(pickle_path), "wb") as file:
            pickle.dump(self.samples, file)

    @staticmethod
    def combine_multiple(datasets: list["ZeusDataset"]) -> "ZeusDataset":
        """Combines multiple LMX datasets into one"""
        assert len(datasets) > 0

        # only one
        if len(datasets) == 1:
            return datasets[0]

        # combine
        name = ""
        samples: list[ZeusDatasetSample] = []
        for dataset in datasets:
            samples += dataset.samples
            if name != "":
                name += "-and-"
            name += dataset.name
        return ZeusDataset(
            name=name,
            samples=samples,
        )

    def print_statistics(self):
        """Prints dataset statistics into the console"""
        avg_len = np.mean([len(sample.lmx.split()) for sample in self.samples])
        print(
            f"Loaded dataset {self.name}, {len(self.samples)} "
            + f"examples, {avg_len:.2f} avg length."
        )
