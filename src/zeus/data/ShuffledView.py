from .ZeusDataset import ZeusDataset
from .ZeusDatasetSample import ZeusDatasetSample
import random
from typing import Iterable


class ShuffledView:
    """A shuffled view at a ZeusDataset"""
    def __init__(self, dataset: ZeusDataset, index_map: list[int]):
        """The shuffled view should NOT be created via constructor,
        see the provided factory methods instead."""

        self.dataset = dataset
        """The underlying dataset being shuffled"""

        self.index_map: list[int] = index_map
        """Each item is an index in the samples list of the underlying dataset,
        this index map is shuffled randomly to emulate the shuffling
        of the underlying dataset."""

        assert len(self.dataset.samples) == len(self.index_map), \
            "The index map should have the same size as the dataset"

    @staticmethod
    def create_unshuffled_for(dataset: ZeusDataset) -> "ShuffledView":
        """Creates a shuffled view of a dataset that actually
        isn't shuffled at all"""
        return ShuffledView(
            dataset=dataset,
            index_map=list(range(len(dataset.samples))),
        )

    @staticmethod
    def create_random_for(dataset: ZeusDataset, seed: int) -> "ShuffledView":
        """Creates a randomly shuffled view of a dataset"""
        view = ShuffledView.create_unshuffled_for(dataset)
        
        # actually shuffle the view
        random.Random(seed).shuffle(view.index_map)
        
        return view
    
    def iter_shuffled_samples(self) -> Iterable[ZeusDatasetSample]:
        """Returns an interator that emits dataset samples in the shuffled order"""
        for i in range(len(self.dataset.samples)):
            yield self.dataset.samples[self.index_map[i]]
