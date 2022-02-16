from torch.utils.data import Dataset
import random
from typing import (
    Callable,
    Dict,
    Generic,
    Iterable,
    Iterator,
    List,
    Optional,
    Sequence,
    Tuple,
    TypeVar,
)
T_co = TypeVar('T_co', covariant=True)
class Subset_noisy(Dataset[T_co]):
    r"""
    Subset of a dataset at specified indices.

    Args:
        dataset (Dataset): The whole Dataset
        indices (sequence): Indices in the whole set selected for subset
    """
    dataset: Dataset[T_co]
    indices: Sequence[int]

    def __init__(self, dataset: Dataset[T_co], indices: Sequence[int],numclass) -> None:
        self.dataset = dataset
        self.indices = indices
        self.numclass=numclass

    def __getitem__(self, idx):
        #if isinstance(idx, list):
            #new_set = self.dataset[[self.indices[i] for i in idx]] #ToDo not sure when it uses this line
            #for i in idx:
               # img,tar= self.dataset[self.indices[i]]
               # tar=random.randrange(self.numclass)
            #a=1

            #return new_set
        img, tar = self.dataset[idx]
        if idx in self.indices:
            tar = random.randrange(self.numclass)
        return img, tar

    def __len__(self):
        return len(self.dataset)
