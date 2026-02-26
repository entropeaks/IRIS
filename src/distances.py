import numpy as np
from abc import ABC, abstractmethod
from typing import Any, List, Optional
from .models.feature_extractors import FeatureExtractor
import torch

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


class FeatureBasedDistance(DistanceStrategy):

    def __init__(self, feature_extractors: List[FeatureExtractor], weights: np.ndarray[float]):

        if len(feature_extractors) != len(weights):
            raise ValueError(
                f"Mismatch: {len(feature_extractors)} extractors but {len(weights)} weights"
            )
        
        if not np.isclose(weights.sum(), 1.0):
            raise ValueError(f"Weights must sum to 1.0, got {weights.sum()}")
        
        self._feature_extractors = feature_extractors
        self._n_features = len(feature_extractors)
        self._weights = weights


    def compute_distances(
        self, 
        query_features: Any, 
        stored_features: Any
    ) -> np.ndarray:
        
        feature_dists = np.zeros((1, len(stored_features), self._n_features), dtype=np.float32)
        for i in range(len(stored_features)):
            for j in range(self._n_features):
                feature_dists[0, i, j] = self._feature_extractors[j].compute_distances(
                    query_features[j],
                    stored_features[i][j]
                )

        normalized_dists = self._normalize_distances_minmax(feature_dists)
        final_dists = np.einsum('ijk,k->ij', normalized_dists, self._weights)

        return final_dists[0]
    
    
    def compute_distances_batch(
        self,
        query_features: Any,
        stored_features_batch: List[Any]
    ) -> np.ndarray:
        
        feature_dists = []
        print(len(stored_features_batch))
        
        for k in range(self._n_features):
            dists = self._feature_extractors[k].compute_distances_batch(
                [sample[k] for sample in query_features],
                None
                )
            print(dists.shape)
            feature_dists.append(dists)

        #normalized_dists = self._normalize_distances_minmax(feature_dists)

        normalized_dists = np.array(feature_dists)
        print(normalized_dists)

        final_dists = np.einsum('ijk,i->jk', normalized_dists, self._weights)
        print(final_dists)

        return final_dists
    

    def _normalize_distances_minmax(self, dists: np.ndarray) -> np.ndarray:
        max_vals = dists.max(axis=1, keepdims=True)
        min_vals = dists.min(axis=1, keepdims=True)
        dist_range = max_vals - min_vals
        dist_range = np.where(dist_range > 0, dist_range, 1.0)

        return (dists - min_vals) / dist_range