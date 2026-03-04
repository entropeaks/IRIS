import numpy as np
from abc import ABC, abstractmethod
from typing import Any, List, Optional
from .feature_stores import IndexManager
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
    def compute(self, primitives: list) -> np.ndarray:
        pass

class BinaryJaccardKernel(DistanceKernel):

    def __init__(self):
        pass

    def compute(self, primitives: list[csr_matrix]):
        intersections, gallery_sum, query_sum = primitives
        unions = query_sum[:, None] + gallery_sum[None, :] - intersections
        return (intersections / unions).toarray()


class FeatureBasedDistance(DistanceStrategy):

    def __init__(self, kernels: list[DistanceKernel], weights: np.ndarray[float]):

        # Validation des paramètres
        if len(kernels) != len(weights):
            raise ValueError(
                f"Mismatch: {len(kernels)} distance kernels but {len(weights)} weights"
            )
        
        weights_array = np.array(weights, dtype=np.float32)
        if not np.isclose(weights_array.sum(), 1.0):
            raise ValueError(f"Weights must sum to 1.0, got {weights_array.sum()}")
        
        self._kernels = kernels
        self._weights = weights
        self._n_features = len(weights)

    def compute_distances(
        self, 
        query_features: Any, 
        stored_features: Any
    ) -> np.ndarray:
        
        pass

    
    def compute_distances_batch(
        self,
        index_manager: IndexManager,
        query_features: Any
    ) -> np.ndarray:
        
        feature_dists = []
        
        for k in range(self._n_features):
            index = index_manager.get_index(k)
            dist_strategy = self._kernels[k]
            encoded_query = index.encode(query_features[k])
            primitives = index.compute_primitives(encoded_query)
            dists = dist_strategy.compute(primitives)
            feature_dists.append(dists)

        #normalized_dists = self._normalize_distances_minmax(feature_dists)

        normalized_dists = np.array(feature_dists)

        final_dists = np.einsum('ijk,i->jk', normalized_dists, self._weights)
        print(final_dists)

        return final_dists
    

    def _normalize_distances_minmax(self, dists: np.ndarray) -> np.ndarray:
        max_vals = dists.max(axis=1, keepdims=True)
        min_vals = dists.min(axis=1, keepdims=True)
        dist_range = max_vals - min_vals
        dist_range = np.where(dist_range > 0, dist_range, 1.0)

        return (dists - min_vals) / dist_range