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
        pass

    @abstractmethod
    def get_features(self, path_to_img: str) -> List:
        pass
    
    @abstractmethod
    def compute_distances(self, desc1, desc2) -> float:
        pass

    @abstractmethod
    def fit(self, feature_list: List) -> None:
        pass

class CompactSparseOccurenceMatrix():

    def __init__(self):
        self._rows = []
        self._cols = []
        self._n_entries = 0
        self._data = []
        self._vocabulary = {}
        self._csr_matrix = None

    def update_csr(self):
        self._csr_matrix = csr_matrix((self._data, (self._rows, self._cols)),
                             shape=(self._n_entries, len(self._vocabulary)))

    def add(self, entry_features: List[str]):
        if entry_features:
            for word in entry_features:
                self._add_to_vocab(word)
                self._rows.append(self._n_entries)
                self._cols.append(self._vocabulary[word])
                self._data.append(1) #occurrence is not frequency but binary
        
        self._n_entries += 1

    def _add_to_vocab(self, word: str):
        if word not in self._vocabulary:
            self._vocabulary[word] = len(self._vocabulary)

    def get_vocab(self) -> dict:
        return self._vocabulary
    
    #-----------------------------------------------------------------------------------------------
    #TODO decouple distances computation from the matrix representation to allow different distances

    def get_jaccard_distances(self, entry_features: List[str]):
        rows, cols, data, _ = self._generate_sparse_matrix_repr([entry_features])
        entry_vector = csr_matrix((data, (rows, cols)),
                                     shape=(1, len(self._vocabulary)))
        
        if self._csr_matrix == None or self._csr_matrix.get_shape()[1] != len(self._vocabulary):
            self.update_csr()

        intersections = entry_vector.dot(self._csr_matrix.T)
        x_size = entry_vector.getnnz()
        row_sizes = self._csr_matrix.getnnz(axis=1)
        unions = x_size + row_sizes - intersections.A1

        return (intersections.A1 / unions).toarray()
    

    def _generate_sparse_matrix_repr(self, entries_features: List[str]):
        rows = []
        cols = []
        data = []

        for row, entry_features in enumerate(entries_features):
            for word in entry_features:
                if word in self._vocabulary:
                    rows.append(row)
                    cols.append(self._vocabulary[word])
                    data.append(1)

        n_entries = row + 1

        return rows, cols, data, n_entries
    

    def get_jaccard_distances_batch(self, entries_features: List[List[str]]):
        rows, cols, data, n_entries = self._generate_sparse_matrix_repr(entries_features)
        entries_matrix = csr_matrix((data, (rows, cols)),
                                    shape=(n_entries, len(self._vocabulary)))
        
        print(entries_matrix.shape)
        print(self._csr_matrix.shape)

        if self._csr_matrix == None or self._csr_matrix.get_shape()[1] != len(self._vocabulary):
            self.update_csr()
        
        print(self._csr_matrix.shape)

        intersections = entries_matrix.dot(self._csr_matrix.T)
        gallery_sum = self._csr_matrix.sum(axis=1)
        entries_sum = entries_matrix.sum(axis=1)
        unions = entries_sum + gallery_sum.T - intersections

        return (intersections / unions).toarray()


class OCRExtractor(FeatureExtractor):
    
    def __init__(self):
        self._trainable = True
        self._occurence_matrix = CompactSparseOccurenceMatrix()

    def fit(self, feature_list):
        print(feature_list)
        for entry_features in feature_list:
            self._occurence_matrix.add(entry_features)
        self._occurence_matrix.update_csr()

    def compute_distances(self, entry_features: List[str]):
        distances = self._occurence_matrix.get_jaccard_distances(entry_features)
        return distances
    
    def compute_distances_batch(self, entries_features, stored_features):
        distances = self._occurence_matrix.get_jaccard_distances_batch(entries_features)
        return distances
    

class PaddleTextExtractor(FeatureExtractor):

    def __init__(self):
        self.ocr = PaddleOCR(use_doc_orientation_classify=False)

    def get_features(self, path_to_img: str) -> List[str]:
        txts = []
        result = self.ocr.predict(path_to_img)
        for res in result:
            txts.extend(res["rec_texts"])

        return txts
    
    def compute_distances(self, desc1: List[str], desc2: List[str]):
        if len(desc1) == 0 and len(desc2) == 0:
            return 1
        
        total = 0
        count = 0
        for word1 in desc1:
            for word2 in desc2:
                rate = fuzz.ratio(word1, word2)
                if rate != 0:
                    total += rate
                    count += 1

        return 1/(1+(total/count)) if count != 0 else 1
    
class RapidTextExtractor(OCRExtractor):

    def __init__(self):
        super().__init__()
        self.ocr = RapidOCR(params={"Global.log_level": "critical"})

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