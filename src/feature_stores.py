from abc import ABC, abstractmethod
from typing import List, Tuple, Any, Optional
import numpy as np
import sys
from .distances import DistanceStrategy
from itertools import chain
from collections import deque
from sys import getsizeof, stderr


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
    def bulk_add(self, items: List[Tuple[str, np.ndarray, dict]]):
        """Ajoute plusieurs images en batch (plus efficace)."""
        pass
    
    @abstractmethod
    def search(self, query_features: np.ndarray, k: int = 5) -> List[Tuple[str, float]]:
        """
        Cherche les k images les plus proches.
        Returns: List[(image_id, distance)]
        """
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

    def __init__(self,
                 distance_strategy: DistanceStrategy):
        self._features = {}
        self._index = []
        self._distance_strategy = distance_strategy
        
    def add(self, image_id, features):
        self._features[image_id] = features
        self._index.append(image_id)

    def bulk_add(self, items):
        for image_id, features in items:
            self.add(image_id, features)

    def search(self, query_features, k):
        dists = self._distance_strategy.compute_distances(query_features, self.get_feature_gallery())
        indices = np.argsort(dists)[:k]
        return np.array(self._index)[indices]

    def get_feature_gallery(self):
        feature_gallery = [self._features[k] for k in self._index]
        return feature_gallery
    
    def get_paths_gallery(self):
        return self._index.copy()
    
    def get(self, image_id):
        if not image_id in self._features:
            raise KeyError("Image not in the database. Please use add method first.")
        return self._features[image_id]
    
    def size(self):
        return total_size(self._features)
    
    def clear(self):
        self._features = {}
        self._index = []
    
    def __len__(self):
        return len(self._index)