from abc import ABC, abstractmethod
from typing import Tuple, Any, Optional
import numpy as np
from itertools import chain
from collections import deque
from sys import getsizeof, stderr
from src.types import Feature


def total_size(o, handlers={}, verbose=False):
    """ Returns the approximate memory footprint an object and all of its contents.

    Automatically finds the contents of the following builtin containers and
    their subclasses:  tuple, list, deque, dict, set and frozenset.
    To search other containers, add handlers to iterate over their contents:

        handlers = {SomeContainerClass: iter,
                    OtherContainerClass: OtherContainerClass.get_elements}

    """
    dict_handler = lambda d: chain.from_iterable(d.items())
    all_handlers = {tuple: iter,
                    list: iter,
                    deque: iter,
                    dict: dict_handler,
                    set: iter,
                    frozenset: iter,
                   }
    all_handlers.update(handlers)     # user handlers take precedence
    seen = set()                      # track which object id's have already been seen
    default_size = getsizeof(0)       # estimate sizeof object without __sizeof__

    def sizeof(o):
        if id(o) in seen:       # do not double count the same object
            return 0
        seen.add(id(o))
        s = getsizeof(o, default_size)

        if verbose:
            print(s, type(o), repr(o), file=stderr)

        for typ, handler in all_handlers.items():
            if isinstance(o, typ):
                s += sum(map(sizeof, handler(o)))
                break
        return s

    return sizeof(o)


class FeatureStore(ABC):
    """Storage for the per-channel features of a gallery.

    An entry is one image id mapped to a list holding one feature per retrieval
    channel, in channel order. "Blocks" slice that the other way: block `i` is
    channel `i`'s feature for every stored image, which is the shape an index
    is built from.
    """

    @abstractmethod
    def add(self, image_id: str, features: list[Feature]) -> None:
        """Store one feature per channel for `image_id`, replacing any previous entry."""

    @abstractmethod
    def bulk_add(self, image_ids: list[str], blocks: list[list[Feature]]) -> None:
        """Store many entries at once, given per-channel blocks in channel order."""

    @abstractmethod
    def get(self, image_id: str) -> list[Feature]:
        """Every channel's feature for one image."""

    @abstractmethod
    def get_feature_block(self, block_id: int) -> list[Feature]:
        """One channel's feature for every stored image, in insertion order."""

    @abstractmethod
    def get_features_blocks(self) -> list[list[Feature]]:
        """Every channel's block, in channel order."""

    @abstractmethod
    def get_feature_gallery(self) -> list[list[Feature]]:
        """Every entry, in insertion order."""

    @abstractmethod
    def get_paths_gallery(self) -> list[str]:
        """The stored image ids, in the order the blocks follow."""

    @abstractmethod
    def memory_footprint(self) -> int:
        """Approximate bytes held. Use `len(store)` for the number of entries."""

    @abstractmethod
    def clear(self) -> None:
        """Drop everything."""

    @abstractmethod
    def __len__(self) -> int:
        """Number of stored entries."""


class InMemoryStore(FeatureStore):

    def __init__(self):
        self._store = {}
        self._index = []
        
        
    def add(self, image_id, features):

        if self._store:
            expected = len(self._store[next(iter(self._store))])
            if len(features) != expected:
                raise ValueError(f"""Expected {expected} features, got {len(features)}.\n
                Clear the store and start over with the right number of features.""")

        if image_id not in self._store:
            self._index.append(image_id)
        #upsert -> features overwritten even if image_id found in the index
        self._store[image_id] = features


    def bulk_add(self, image_ids, blocks):
        for i, image_id in enumerate(image_ids):
            features = [block[i] for block in blocks]
            self.add(image_id, features)


    def get_feature_gallery(self):
        feature_gallery = [self._store[k] for k in self._index]
        return feature_gallery
    
    
    def get_feature_block(self, block_id: int) -> list:
        return [self._store[k][block_id] for k in self._index]
    

    def get_features_blocks(self) -> list[list]:
        if len(self._store) == 0:
            raise LookupError("Empty feature store. Please use add method first.")
        feature_num = len(self._store[next(iter(self._store))])
        return [self.get_feature_block(i) for i in range(feature_num)]
    
    
    def get_paths_gallery(self):
        return self._index.copy()
    
    
    def get(self, image_id):
        if not image_id in self._store:
            raise KeyError("Image not in the database. Please use add method first.")
        return self._store[image_id]
    
    
    def memory_footprint(self):
        return total_size(self._store)
    
    
    def clear(self):
        self._store = {}
        self._index = []
    

    def __len__(self):
        return len(self._index)
    

    def __getitem__(self, idx):
        if isinstance(idx, list) or isinstance(idx, np.ndarray):
            return [self._index[i] for i in idx]
        return self._index[idx]