from abc import ABC, abstractmethod
import warnings
from scipy.sparse import csr_matrix, vstack
import numpy as np
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

    def __init__(self, kernel, weighting_strategy: WeightingStrategy,
                 staleness_threshold: float=0.2):
        super().__init__(kernel)
        self._weighting_strategy = weighting_strategy
        self._vocabulary = {}
        self._gallery: csr_matrix | None = None
        self._staleness_threshold = staleness_threshold
        self._reset_staleness()

    def _reset_staleness(self) -> None:
        self._rows_at_build = 0
        self._rows_added = 0
        self._terms_seen = 0
        self._terms_dropped = 0
        self._staleness_warned = False

    @property
    def staleness(self) -> dict:
        """How far the index has drifted from the corpus its vocabulary describes.

        `added_ratio` is entries appended since the last build, over the entries
        that build saw. `oov_ratio` is the share of term occurrences in those
        entries that fell outside the vocabulary and were dropped. Both grow
        monotonically until `build` runs again; the index cannot rebuild itself,
        since it keeps only the transformed matrix.
        """
        return {"rows_at_build": self._rows_at_build,
                "rows_added": self._rows_added,
                "added_ratio": self._rows_added / max(self._rows_at_build, 1),
                "oov_ratio": self._terms_dropped / max(self._terms_seen, 1)}

    def build(self, gallery_features):
        raw = self._build_raw_gallery(gallery_features)
        self._gallery = self._weighting_strategy.fit_transform(raw)
        self._is_empty = False
        self._reset_staleness()
        self._rows_at_build = raw.shape[0]

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


    def clear(self) -> None:
        """Forget the gallery and the vocabulary built from it.

        The base implementation drops only the matrix; keeping the vocabulary
        would make a rebuild encode the new corpus against the old one's words.
        """
        super().clear()
        self._vocabulary = {}
        self._reset_staleness()


    def _add_to_vocab(self, word: str):
        if word not in self._vocabulary:
            self._vocabulary[word] = len(self._vocabulary)
        

    def update(self, new_gallery_features: list[Feature]) -> None:
        """Append entries to the index, encoded against the vocabulary built earlier.

        Words absent from that vocabulary are dropped, the same rule `encode`
        already applies to queries: growing the vocabulary would mean rebuilding
        every stored row and refitting the weighting, which is `build`'s job.
        """
        if self.is_empty():
            return self.build(new_gallery_features)

        raw, terms, dropped = self._encode_counting(new_gallery_features)

        unreachable = int((raw.getnnz(axis=1) == 0).sum())
        if unreachable:
            warnings.warn(
                f"{unreachable} of {raw.shape[0]} added entries hold no word from the "
                f"index vocabulary. They are stored as empty rows and can never be "
                f"retrieved, not even by their own words; rebuild the index to admit "
                f"their terms", RuntimeWarning, stacklevel=2)

        self._rows_added += raw.shape[0]
        self._terms_seen += terms
        self._terms_dropped += dropped
        self._gallery = vstack([self._gallery, self._weighting_strategy.transform(raw)],
                               format="csr")
        self._warn_if_stale()
    

    def encode(self, query_features: list[list[str]]) -> csr_matrix:
        raw, _, _ = self._encode_counting(query_features)
        return self._weighting_strategy.transform(raw)

    def _warn_if_stale(self) -> None:
        """Warn once when the index has drifted far enough to deserve a rebuild."""
        if self._staleness_warned:
            return
        drift = self.staleness
        if max(drift["added_ratio"], drift["oov_ratio"]) <= self._staleness_threshold:
            return
        warnings.warn(
            f"index is drifting from its vocabulary: {drift['added_ratio']:.0%} of the "
            f"corpus added since build, {drift['oov_ratio']:.0%} of their terms dropped. "
            f"Weights still reflect the corpus build saw; rebuild to refresh them",
            RuntimeWarning, stacklevel=3)
        self._staleness_warned = True


    def _encode_counting(self, query_features: list[list[str]]) -> tuple[csr_matrix, int, int]:
        """Encode against the fixed vocabulary, also reporting terms seen and dropped."""
        rows = []
        cols = []
        data = []
        seen = dropped = 0

        for row, entry_features in enumerate(query_features):
            for word in entry_features:
                seen += 1
                if word in self._vocabulary:
                    rows.append(row)
                    cols.append(self._vocabulary[word])
                    data.append(1)
                else:
                    dropped += 1

        n_entries = len(query_features)

        query_matrix = csr_matrix((data, (rows, cols)),
                                    shape=(n_entries, len(self._vocabulary)))

        return query_matrix, seen, dropped