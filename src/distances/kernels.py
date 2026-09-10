import numpy as np
from abc import ABC, abstractmethod
from typing import Any, List, Optional
from src.types import Matrix
import torch
from scipy.sparse import csr_matrix

from src.types import DistanceKernel


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

    def pairwise(self, query: np.ndarray, gallery: np.ndarray) -> torch.Tensor:
        dists = torch.cdist(torch.from_numpy(query), torch.from_numpy(gallery), p=2)

        return dists


class BhattacharyyaKernel(DistanceKernel):

    def preprocess(self, query: np.ndarray) -> np.ndarray:
        return np.sqrt(query)
    
    def pairwise(self, query: np.ndarray, gallery: np.ndarray):
        similarity = query.dot(gallery.T)
        similarity = np.clip(similarity, 1e-10, 1.0)
        distances = -np.log(similarity)

        return distances