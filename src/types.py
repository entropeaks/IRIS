from scipy.sparse import csr_matrix
from numpy import ndarray
from torch.utils.data import DataLoader
from typing import TypeAlias, Any, Optional
from dataclasses import dataclass
from abc import ABC, abstractmethod

Matrix: TypeAlias = csr_matrix | ndarray
Feature: TypeAlias = int | float | list[Any] | ndarray

class FeatureExtractor(ABC):

    def __init__(self):
        self.trainable: bool

    @abstractmethod
    def get_features(self, imgs_arrays_rgb: list[ndarray]) -> list[list[Feature]]:
        pass

    @abstractmethod
    def fit(self, dataloader: DataLoader):
        pass


class DistanceKernel(ABC):
    
    @abstractmethod
    def pairwise(self, query, gallery) -> ndarray:
        pass


class BaseIndex(ABC):
    def __init__(self, kernel: DistanceKernel):
        self._kernel = kernel
        self._gallery: list | None = None
        self._is_empty: bool=True

    @abstractmethod
    def build(self, gallery_features: list[Feature]) -> None: ...
    
    @abstractmethod
    def update(self, features: list[Feature]) -> None: ...

    @abstractmethod
    def encode(self, query_features: list[Feature]) -> Matrix: ...
    
    def distances(self, query_features: list[Feature]) -> ndarray:
        encoded_query = self.encode(query_features)
        return self._kernel.pairwise(encoded_query, self._gallery)
    
    def is_empty(self) -> bool:
        return self._is_empty

    def clear(self) -> None:
        self._gallery = None
        self._is_empty = True

@dataclass
class RetrievalChannel:
    extractor: FeatureExtractor
    index: BaseIndex
    kernel: DistanceKernel
    weight: Optional[float] = None
    is_trainable: bool = False