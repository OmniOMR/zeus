import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

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

    # There used to be a `musicxml` field here, carried so that MusicXML-level
    # evaluation could read it; that evaluation belongs to a benchmarking rig
    # rather than to this repository, and nothing here ever read the field.
    # Pickles written while it existed still load — pickle restores a dataclass
    # through its `__dict__`, so the stale attribute simply rides along and is
    # ignored. That would stop being true if this became a slots dataclass.

    def __postinit__(self):
        assert Path(self.sample_name).as_posix() == self.sample_name
        assert len(self.image) > 0
        assert "\n" not in self.lmx
        assert "\r" not in self.lmx


class _Unpickler(pickle.Unpickler):
    """Reads dataset pickles, including ones written before the modules moved.

    A pickle stores the import path of every class it contains, so renaming a
    module invalidates every pickle that mentions it — and these hold decoded
    images, so rebuilding one is minutes of work per dataset split, not
    seconds. `ZeusDatasetSample` used to live in a module of its own and now
    sits beside `ZeusDataset`, which is enough to make an existing pickle fail
    to load with `ModuleNotFoundError`.

    Redirecting the old path here keeps those files readable. Nothing writes
    the old path any more, so this table only ever shrinks.
    """

    MOVED_MODULES = {
        "zeus.data.ZeusDatasetSample": "zeus.data.zeus_dataset",
        "zeus.data.ZeusDataset": "zeus.data.zeus_dataset",
    }

    def find_class(self, module: str, name: str) -> Any:
        return super().find_class(self.MOVED_MODULES.get(module, module), name)


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

                zeus_dataset_samples.append(
                    ZeusDatasetSample(
                        sample_name=sample.name,
                        image=image,
                        lmx=lmx,
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
            samples = _Unpickler(file).load()
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
