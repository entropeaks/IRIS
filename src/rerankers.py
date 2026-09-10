import cv2
from abc import ABC, abstractmethod

import numpy as np
from torch.utils.data import Dataset

from src.extractors import FeatureExtractor, OrbFeatureExtractor, HSVExtractor
from src.distances.kernels import DistanceKernel, BhattacharyyaKernel


class Reranker(ABC):

    @abstractmethod
    def _pairwise():
        pass
    
    @abstractmethod
    def score():
        pass

class HSVReranker(Reranker):

    def __init__(self, hsv: HSVExtractor=HSVExtractor(), kernel: DistanceKernel=BhattacharyyaKernel()):
        self.extractor = hsv
        self.kernel = kernel

    def score(self, query_imgs: Dataset, gallery_imgs: Dataset, candidates_indices: np.ndarray) -> np.ndarray:
        nQ, top_k = candidates_indices.shape
        dists = np.full((nQ, top_k), np.inf)
        for query_idx in range(nQ):
            ref = query_imgs[query_idx][0]
            ref_des = self.extractor.get_features([ref])[0]
            for i, idx in enumerate(candidates_indices[query_idx]):
                comp = gallery_imgs[idx][0]
                comp_des = self.extractor.get_features([comp])[0]
                dist = self._pairwise(ref_des.ravel(), comp_des.ravel())
                dists[query_idx, i] = dist

        return dists

    def _pairwise(self, ref1: np.ndarray, ref2: np.ndarray):
        ref1 = self.kernel.preprocess(ref1)
        ref2 = self.kernel.preprocess(ref2)
        return self.kernel.pairwise(ref1, ref2)
        


class ORBReranker(Reranker):

    def __init__(self, orb: OrbFeatureExtractor=OrbFeatureExtractor()):
        self.extractor = orb
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)


    def score(self, query_imgs: Dataset, gallery_imgs: Dataset, candidates_indices: np.ndarray) -> np.ndarray:
        nQ, top_k = candidates_indices.shape
        dists = np.full((nQ, top_k), np.inf)
        for query_idx in range(nQ):
            ref = query_imgs[query_idx][0]
            ref_des = self.extractor.get_features([ref])[0]
            for i, idx in enumerate(candidates_indices[query_idx]):
                comp = gallery_imgs[idx][0]
                comp_des = self.extractor.get_features([comp])[0]
                dist = self._pairwise(ref_des, comp_des)
                dists[query_idx, i] = dist

        return dists


    def _pairwise(self, des1: str, des2: str) -> float:
        if des1 is None or des2 is None:
            return float('inf')
        
        matches = self.bf.match(des1, des2)

        if not matches:
            return float('inf')

        avg_distance = sum(m.distance for m in matches) / len(matches)

        return avg_distance