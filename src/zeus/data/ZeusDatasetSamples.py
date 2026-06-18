from pathlib import Path
from typing import overload
from .ZeusDatasetSample import ZeusDatasetSample


class ZeusDatasetSamples:
    """Represents a samples file for a dataset, e.g. 'samples.train.txt'"""
    
    def __init__(self, file_path: Path):
        self.file_path: Path = file_path
        """
        Path to the samples file.
        It may not exist, only points to where it should be.
        For loaded samples, the file should exist, for just being created
        samples, the file may not yet exist.
        """

        self.__samples: list[ZeusDatasetSample] = []
        
    
    def append(self, sample_name: str) -> ZeusDatasetSample:
        """
        Appends a sample to the list of samples and returns the new sample.
        
        Example sample name:
        `samples/chopin/mazurkas/mazurka17-2/maj2_down_m-0-3`
        """
        sample = ZeusDatasetSample(
            name=sample_name,
            path=self.file_path.parent / sample_name
        )
        self.__samples.append(sample)
        return sample
    
    def __len__(self) -> int:
        return len(self.__samples)
    
    def __iter__(self):
        yield from self.__samples
    
    @overload
    def __getitem__(self, index: int) -> ZeusDatasetSample:
        pass

    @overload
    def __getitem__(self, index: slice) -> "ZeusDatasetSamples":
        pass

    def __getitem__(self, index: int | slice):
        if isinstance(index, int):
            return self.__samples[index]
        elif isinstance(index, slice):
            subset = ZeusDatasetSamples(
                self.file_path.with_name(
                    self.file_path.name
                    + f"-slice_{index.start}_{index.stop}_{index.step}"
                )
            )
            subset.__samples = self.__samples[index]
            return subset
        else:
            raise TypeError("Unknown argument type")
    
    def write(self):
        """Writes the samples file to disk"""
        with open(self.file_path, "w") as f:
            for sample in self:
                f.write(sample.name + "\n")

    @staticmethod
    def load(file_path: Path) -> "ZeusDatasetSamples":
        """Loads a samples file"""
        samples = ZeusDatasetSamples(file_path)
        with open(file_path, "r") as f:
            for line in f.readlines():
                samples.append(line.strip())
        return samples
    
    @staticmethod
    def empty(file_path: Path) -> "ZeusDatasetSamples":
        """Creates a new and empty samples file (in-memory before written)"""
        return ZeusDatasetSamples(file_path)
