from abc import ABC, abstractmethod
from typing import Tuple, Any, Optional, TypeAlias
import numpy as np
import sys
from itertools import chain
from collections import deque
from sys import getsizeof, stderr
from scipy.sparse import csr_matrix

Feature: TypeAlias = int | float | list[Any]


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
    
    """ @abstractmethod
    def search(self, query_features: np.ndarray, k: int = 5) -> list[Tuple[str, float]]:
        "
        Cherche les k images les plus proches.
        Returns: list[(image_id, distance)]
        "
        pass """
    
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
        self._features_by_id = {}
        self._features_blocks = []
        self._index = []
        
    def add(self, image_id, features):
        self._features_by_id[image_id] = features

        if len(self._features_blocks) == 0:
            self._features_blocks = [[] for _ in range(len(features))]

        for i, feat in enumerate(features):
            self._features_blocks[i].append(feat)

        self._index.append(image_id)

    def bulk_add(self, items):
        for image_id, features in items:
            self.add(image_id, features)

    """ def search(self, query_features, k):
        dists = self._distance_strategy.compute_distances(query_features, self.get_feature_gallery())
        indices = np.argsort(dists)[:k]
        return np.array(self._index)[indices] """

    def get_feature_gallery(self):
        feature_gallery = [self._features_by_id[k] for k in self._index]
        return feature_gallery
    
    def get_features_blocks(self):
        return self._features_blocks
    
    def get_paths_gallery(self):
        return self._index.copy()
    
    def get(self, image_id):
        if not image_id in self._features_by_id:
            raise KeyError("Image not in the database. Please use add method first.")
        return self._features_by_id[image_id]
    
    def size(self):
        return total_size(self._features_by_id)
    
    def clear(self):
        self._features_by_id = {}
        self._index = []
    
    def __len__(self):
        return len(self._index)
    

class BaseIndex(ABC):

    @abstractmethod
    def build(self, gallery_features: list[Feature]) -> None:
        pass
    
    @abstractmethod
    def update(self, entry_features: list[Feature]) -> None:
        pass

    @abstractmethod
    def encode(self, entry_features: list[Feature]) -> csr_matrix:
        pass

    @abstractmethod
    def compute_primitives(self, encoded_query: csr_matrix) -> list:
        pass


class TextCSRIndex(BaseIndex):

    def __init__(self):
        self._rows = []
        self._cols = []
        self._n_entries = 0
        self._data = []
        self._vocabulary = {}
        self._csr_matrix = None

    def build(self, gallery_features: list[Feature]):
        for entry in gallery_features:
            self._add(entry)
        
        self._update_csr()

    def _add(self, entry_features: list[str]):
        if entry_features:
            for word in entry_features:
                self._add_to_vocab(word)
                self._rows.append(self._n_entries)
                self._cols.append(self._vocabulary[word])
                self._data.append(1) #occurrence is not frequency but binary
        
        self._n_entries += 1
        

    def _add_to_vocab(self, word: str):
        if word not in self._vocabulary:
            self._vocabulary[word] = len(self._vocabulary)


    def _update_csr(self):
        self._csr_matrix = csr_matrix((self._data, (self._rows, self._cols)),
                             shape=(self._n_entries, len(self._vocabulary)))
        
    def update(self):
        pass

    def encode(self, query_features: list[str]) -> csr_matrix:
        rows = []
        cols = []
        data = []

        for row, entry_features in enumerate(query_features):
            for word in entry_features:
                if word in self._vocabulary:
                    rows.append(row)
                    cols.append(self._vocabulary[word])
                    data.append(1)

        n_entries = len(entry_features)

        query_matrix = csr_matrix((data, (rows, cols)),
                                    shape=(n_entries, len(self._vocabulary)))

        return query_matrix
    

    def compute_primitives(self, query_matrix: csr_matrix):
        intersections = self._get_intersection_counts(query_matrix)
        gallery_sum, entries_sum = self._get_rows_sizes(query_matrix)

        return intersections, gallery_sum, entries_sum
    
    
    def _get_intersection_counts(self, query_matrix: csr_matrix):
        intersect_counts = query_matrix.dot(self._csr_matrix.T)

        return intersect_counts


    def _get_rows_sizes(self, query_matrix: csr_matrix):
        gallery_sum = self._csr_matrix.sum(axis=1)
        entries_sum = query_matrix.sum(axis=1)

        gallery_sum = np.asarray(gallery_sum).ravel()
        entries_sum = np.asarray(entries_sum).ravel()
        
        return gallery_sum, entries_sum


class IndexManager():

    def __init__(self, index: list[BaseIndex]):
        self._index = index

    def get_index(self, i: int) -> BaseIndex:
        return self._index[i]
    
    def build(self, features_blocks: list[Feature]):
        for i, index in enumerate(self._index):
            index.build(features_blocks[i])