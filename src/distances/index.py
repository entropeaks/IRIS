from abc import ABC, abstractmethod
from scipy.sparse import csr_matrix
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.preprocessing import normalize
from src.distances.kernels import DistanceKernel

from src.types import Feature
from src.types import BaseIndex, DistanceKernel


class DenseIndex(BaseIndex):

    def __init__(self, kernel: DistanceKernel):
        super().__init__(kernel)
        self._gallery = []
        self._kernel = kernel

    def build(self, gallery_features: list[Feature]):
        stacked = np.vstack([f.ravel() for f in gallery_features])
        self._gallery = self._kernel.preprocess(stacked)
        self._is_empty = False

    def update(self, new_gallery_features: list[Feature]):
        stacked = np.vstack([f.ravel() for f in new_gallery_features])
        preprocessed = self._kernel.preprocess(stacked)
        self._gallery = np.concat((self._gallery, preprocessed), axis=0) # TODO: modify soon, won't scale

    def encode(self, query_features: np.ndarray) -> np.ndarray:
        stacked = np.vstack([f.ravel() for f in query_features])
        return self._kernel.preprocess(stacked)


class WeightingStrategy(ABC):

    @abstractmethod
    def fit_transform(self, raw_counts: csr_matrix) -> csr_matrix: ...
    
    @abstractmethod
    def transform(self, raw_counts: csr_matrix) -> csr_matrix: ...


class BinaryStrategy(WeightingStrategy):

    def fit_transform(self, raw_counts: csr_matrix):
        return self._binarize(raw_counts)
    
    def transform(self, raw_counts: csr_matrix):
        return self._binarize(raw_counts)
    
    def _binarize(self, raw_counts: csr_matrix):
        return (raw_counts > 0).astype(int)
    
    
class TFIDFStrategy(WeightingStrategy):

    def __init__(self):
        self._idf: np.ndarray | None = None
    
    def fit_transform(self, raw_counts: csr_matrix):
        N = raw_counts.shape[0]
        nt = np.asarray((raw_counts > 0).sum(axis=0)).ravel()
        self._idf = np.log((1 + N) / (1 + nt)) + 1
        return self._apply(raw_counts)
    
    def transform(self, raw_counts: csr_matrix) -> csr_matrix:
        if self._idf is None:
            raise RuntimeError("Strategy must be fit before transform")
        return self._apply(raw_counts)
    
    def _apply(self, raw_counts: csr_matrix) -> csr_matrix:
        tf = normalize(raw_counts, norm='l1', axis=1)
        return tf.multiply(self._idf).tocsr()
    

class SparseIndex(BaseIndex):

    def __init__(self, kernel, weighting_strategy: WeightingStrategy):
        super().__init__(kernel)
        self._weighting_strategy = weighting_strategy
        self._vocabulary = {}
        self._gallery: csr_matrix | None = None

    def build(self, gallery_features):
        raw = self._build_raw_gallery(gallery_features)
        self._gallery = self._weighting_strategy.fit_transform(raw)
        self._is_empty = False

    def _build_raw_gallery(self, gallery_features: list[Feature]):
        rows = []
        cols = []
        data = []
        n_entries = 0

        for entry_features in gallery_features:
            if entry_features:
                for word in entry_features: # stores frequencies, not binary occurrence
                    self._add_to_vocab(word)
                    rows.append(n_entries)
                    cols.append(self._vocabulary[word])
                    data.append(1)

            n_entries += 1

        return csr_matrix((data, (rows, cols)),
                             shape=(n_entries, len(self._vocabulary)))


    def _add_to_vocab(self, word: str):
        if word not in self._vocabulary:
            self._vocabulary[word] = len(self._vocabulary)
        

    def update(self):
        raise NotImplementedError("This method hasn't been implemented yet.")
    

    def encode(self, query_features: list[list[str]]) -> csr_matrix:
        raw = self._raw_encode_with_fixed_vocab(query_features)
        return self._weighting_strategy.transform(raw)


    def _raw_encode_with_fixed_vocab(self, query_features: list[list[str]]) -> csr_matrix:
        rows = []
        cols = []
        data = []

        for row, entry_features in enumerate(query_features):
            for word in entry_features:
                if word in self._vocabulary:
                    rows.append(row)
                    cols.append(self._vocabulary[word])
                    data.append(1)

        n_entries = len(query_features)

        query_matrix = csr_matrix((data, (rows, cols)),
                                    shape=(n_entries, len(self._vocabulary)))

        return query_matrix