from abc import ABC, abstractmethod
from typing import List, Tuple
from paddleocr import PaddleOCR
from rapidocr import RapidOCR
from thefuzz import fuzz
import numpy as np
import cv2
from scipy.sparse import csr_matrix

class FeatureExtractor(ABC):

    def __init__(self):
        self.trainable: bool

    @abstractmethod
    def get_features(self, path_to_img: str) -> List:
        pass
    

class PaddleTextExtractor(FeatureExtractor):

    def __init__(self):
        self.ocr = PaddleOCR(use_doc_orientation_classify=False)
        self.trainable = False

    def get_features(self, path_to_img: str) -> List[str]:
        txts = []
        result = self.ocr.predict(path_to_img)
        for res in result:
            txts.extend(res["rec_texts"])

        return [word.lower() for string in txts for word in string.split(" ")]
    

class RapidTextExtractor(FeatureExtractor):

    def __init__(self):
        super().__init__()
        self.ocr = RapidOCR(params={"Global.log_level": "critical"})
        self.trainable = False

    def get_features(self, path_to_img: str) -> List[str]:
        result = self.ocr(path_to_img)
        if result.txts == None:
            return []
        
        words = [word.lower() for string in result.txts for word in string.split(" ")]
        
        return words
    


class OrbFeatureExtractor(FeatureExtractor):

    def __init__(self, n_features: int=500):
        self.orb = cv2.ORB_create(nfeatures=n_features)
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    def get_features(self, path_to_img: str):
        img = cv2.imread(path_to_img, cv2.IMREAD_GRAYSCALE)
        _, descriptors = self.orb.detectAndCompute(img, None)

        return descriptors
    
    def compute_distances(self, des1: str, des2: str) -> float:
        if des1 is None or des2 is None:
            return float('inf')
        
        matches = self.bf.match(des1, des2)

        if not matches:
            return float('inf')

        avg_distance = sum(m.distance for m in matches) / len(matches)

        return avg_distance
    

class SIFTFeatureExtractor(FeatureExtractor):

    def __init__(self, min_match_count: int=10):
        self.sift = cv2.SIFT_create()
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        self.flann = cv2.FlannBasedMatcher(index_params, search_params)
        self.min_match_count = min_match_count
    
    def get_features(self, path_to_img: str):
        img = cv2.imread(path_to_img, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return None, None
        
        kp, des = self.sift.detectAndCompute(img, None)

        return (kp, des)
    
    def compute_distances(self, feat1: Tuple, feat2: Tuple) -> int: 
        kp1, des1 = feat1
        kp2, des2 = feat2

        if des1 is None or des2 is None or len(des1) < 2 or len(des2) < 2:
            return 1.0

        matches = self.flann.knnMatch(des1, des2, k=2)

        good_matches = []
        for m, n in matches:
            if m.distance < 0.7*n.distance:
                good_matches.append(m)

        if len(good_matches) > self.min_match_count:
            src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1,1,2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1,1,2)

            mask: np.ndarray
            _, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            
            if mask is None:
                return 1.0
            
            inliers_count = np.sum(mask)
            
            return 1.0 / (inliers_count + 1)

        return 1.0