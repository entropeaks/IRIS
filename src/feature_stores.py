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
    """
    Interface abstraite pour le stockage de features.
    Permet de swap entre différents backends (RAM, DB, Faiss).
    """
    
    @abstractmethod
    def add(self, image_id: str, features: Any):
        """Ajoute une image et ses features au store."""
        pass
    
    @abstractmethod
    def bulk_add(self, items: list[Tuple[str, np.ndarray, dict]]):
        """Ajoute plusieurs images en batch (plus efficace)."""
        pass

    @abstractmethod
    def get_feature_block(self) -> list:
        pass

    @abstractmethod
    def get_features_blocks(self) -> list:
        pass
    
    @abstractmethod
    def get(self, image_id: str) -> Optional[np.ndarray]:
        """Récupère les features d'une image."""
        pass
    
    @abstractmethod
    def size(self) -> int:
        """Retourne le nombre d'images dans le store."""
        pass
    
    @abstractmethod
    def clear(self):
        """Vide le store."""
        pass
    

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

    def bulk_add(self, items):
        for image_id, features in items:
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
    
    def size(self):
        return total_size(self._store)
    
    def clear(self):
        self._store = {}
        self._index = []
    
    def __len__(self):
        return len(self._index)