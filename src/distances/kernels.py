import numpy as np
from abc import ABC, abstractmethod
from typing import Any, List, Optional
from src.types import Matrix
import torch
from scipy.sparse import csr_matrix


class DistanceStrategy(ABC):
    """
    Stratégie pour calculer la distance entre features.
    Chaque modèle fournit sa propre stratégie.
    """
    
    @abstractmethod
    def compute_distances(
        self, 
        query_features: Any, 
        stored_features: Any
    ) -> float:
        """Calcule la distance entre query et stored."""
        pass
    
    @abstractmethod
    def compute_distances_batch(
        self,
        query_features: Any,
        stored_features_batch: List[Any]
    ) -> np.ndarray:
        """
        Calcule les distances entre query et un batch.
        Optimisé pour la vectorisation.
        """
        pass


class VectorBasedDistance(DistanceStrategy):

    def __init__(self):
        pass

    def compute_distances(
        self, 
        query_features: torch.Tensor, 
        stored_features: np.ndarray
    ) -> np.ndarray:
        dists = torch.cdist(query_features.unsqueeze(0), torch.stack(stored_features), p=2)
        return dists.numpy()
    
    def compute_distances_batch(
        self,
        query_features: torch.Tensor,
        stored_features_batch: torch.Tensor
    ) -> torch.Tensor:
        """
        Calcule les distances entre query et un batch.
        Optimisé pour la vectorisation.
        """
        dists = torch.cdist(query_features, stored_features_batch, p=2)
        return dists


class DistanceKernel(ABC):
    
    @abstractmethod
    def pairwise(self, query, gallery) -> np.ndarray:
        pass

class BinaryJaccardKernel(DistanceKernel):

    def _get_intersection_counts(self, query: csr_matrix, gallery: csr_matrix):
        intersect_counts = query.dot(gallery.T)

        return intersect_counts

    def _get_rows_sizes(self, query: csr_matrix, gallery: csr_matrix):
        gallery_sum = gallery.sum(axis=1)
        query_sum = query.sum(axis=1)

        gallery_sum = np.asarray(gallery_sum).ravel()
        query_sum = np.asarray(query_sum).ravel()
        
        return query_sum, gallery_sum

    def pairwise(self, query: csr_matrix, gallery: csr_matrix) -> np.ndarray:
        intersect_counts = self._get_intersection_counts(query, gallery)
        query_sum, gallery_sum = self._get_rows_sizes(query, gallery)
        intersections_dense = intersect_counts.toarray()
        unions = query_sum[:, None] + gallery_sum[None, :] - intersections_dense
        jaccard = np.divide(
            intersections_dense, 
            unions, 
            out=np.zeros_like(unions, dtype=float), 
            where=unions != 0                       
        )
        return 1 - jaccard
    
    
class EuclidianDistanceKernel(DistanceKernel):

    def preprocess(self):
        x = x / np.maximum(x.sum(axis=1, keepdims=True), 1e-12)
        return np.sqrt(x)

    def pairwise(self, gallery: torch.Tensor, query: torch.Tensor) -> torch.Tensor:
        dists = torch.cdist(query, gallery, p=2)


class BhattacharyyaKernel(DistanceKernel):

    def preprocess(self, batch: np.ndarray):
        return np.sqrt(batch)
    
    def pairwise(self, preprocessed_query: np.ndarray, preprocessed_gallery: np.ndarray):
        similarity = preprocessed_query.dot(preprocessed_gallery.T)
        similarity = np.clip(similarity, 1e-10, 1.0)
        distances = -np.log(similarity)

        return distances