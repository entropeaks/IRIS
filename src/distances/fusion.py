from src.distances.index import IndexManager
from src.distances.kernels import DistanceKernel
from typing import Any
import numpy as np
from scipy.stats import rankdata


class FeatureBasedFusion():

    def __init__(self, kernels: list[DistanceKernel], weights: list[float]):

        # Validation des paramètres
        if len(kernels) != len(weights):
            raise ValueError(
                f"Mismatch: {len(kernels)} distance kernels but {len(weights)} weights"
            )
        
        weights_array = np.array(weights, dtype=np.float32)
        if not np.isclose(weights_array.sum(), 1.0):
            raise ValueError(f"Weights must sum to 1.0, got {weights_array.sum()}")
        
        self._kernels = kernels
        self._weights = np.array(weights, dtype=np.float64)
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
            dists = index.similarity(query_features[k])
            print(dists)
            print(f"Feature {k}: min={dists.min():.4f}, max={dists.max():.4f}, "
            f"std={dists.std():.4f}, mean={dists.mean():.4f}")
            normalized_dists = self._normalize_distances(dists)
            print(f"Feature {k} after norm: min={normalized_dists.min():.4f}, max={normalized_dists.max():.4f}, "
            f"std={normalized_dists.std():.4f}, mean={normalized_dists.mean():.4f}")
            feature_dists.append(normalized_dists)
        
        final_dists = np.einsum('ijk,i->jk', np.array(feature_dists), self._weights)
        print(final_dists)

        return final_dists
    

    def _normalize_distances(self, dists: np.ndarray) -> np.ndarray:
        return np.apply_along_axis(lambda row: rankdata(row) / len(row), axis=1, arr=dists)
    
        max_vals = dists.max(axis=1, keepdims=True)
        min_vals = dists.min(axis=1, keepdims=True)
        dist_range = max_vals - min_vals
        dist_range = np.where(dist_range > 0, dist_range, 1.0)

        return (dists - min_vals) / dist_range