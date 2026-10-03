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

class SIFTReranker(Reranker):
    """Geometric verification with SIFT, which ORB and SuperPoint cannot do.

    ORB's orientation comes from an intensity centroid and SuperPoint's
    detector was trained upright; both degrade as soon as a cap is turned, and
    on caps that land at arbitrary angles that is every pair. SIFT assigns each
    keypoint a canonical orientation from its own gradient histogram and
    describes the patch in that frame, so the descriptor is the same whichever
    way the cap fell.

    The score is the number of inliers to a similarity transform, not the
    number of matches. Raw match counts are dominated by the ratio test's
    leftovers -- that is what makes an unverified matcher look random. Four
    degrees of freedom (rotation, scale, translation) is what actually relates
    two photographs of the same flat cap; a homography's eight fit the noise as
    happily as the cap.
    """

    def __init__(self, sift: "SIFTFeatureExtractor"=None, ratio: float=0.75,
                 min_match_count: int=8, ransac_threshold: float=5.0):
        from src.extractors import SIFTFeatureExtractor
        self.extractor = sift if sift is not None else SIFTFeatureExtractor()
        self.ratio = ratio
        self.min_match_count = min_match_count
        self.ransac_threshold = ransac_threshold

    def score(self, query_imgs: Dataset, gallery_imgs: Dataset, candidates_indices: np.ndarray) -> np.ndarray:
        nQ, top_k = candidates_indices.shape
        dists = np.full((nQ, top_k), np.inf)
        # candidate lists overlap across queries, so detection is cached
        gallery_features = {}
        for query_idx in range(nQ):
            ref_feat = self.extractor.detect(query_imgs[query_idx][0])
            for i, idx in enumerate(candidates_indices[query_idx]):
                idx = int(idx)
                if idx not in gallery_features:
                    gallery_features[idx] = self.extractor.detect(gallery_imgs[idx][0])
                dists[query_idx, i] = self._pairwise(ref_feat, gallery_features[idx])

        return dists

    def _pairwise(self, feat1: tuple, feat2: tuple) -> float:
        kp1, des1 = feat1
        kp2, des2 = feat2

        if des1 is None or des2 is None or len(des1) < 2 or len(des2) < 2:
            return 1.0

        matcher = cv2.BFMatcher(cv2.NORM_L2)
        good = [m for m, n in matcher.knnMatch(des1, des2, k=2)
                if m.distance < self.ratio * n.distance]

        if len(good) < self.min_match_count:
            return 1.0

        src = np.float32([kp1[m.queryIdx].pt for m in good])
        dst = np.float32([kp2[m.trainIdx].pt for m in good])
        _, mask = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC,
                                              ransacReprojThreshold=self.ransac_threshold)
        if mask is None:
            return 1.0

        return 1.0 / (int(mask.sum()) + 1)
