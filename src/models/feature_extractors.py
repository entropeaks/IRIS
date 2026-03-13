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


class HSVExtractor(FeatureExtractor):

    def __init__(self, method=cv2.HISTCMP_BHATTACHARYYA):
        self.trainable = False
        self._method = method

    def _apply_gray_world(self, rgb_img):
        """
        Applique l'algorithme Gray World pour annuler la dominante de couleur de l'éclairage.
        Utilise des opérations matricielles pour l'optimisation des performances.
        """
        # Conversion en float32 pour éviter les dépassements (overflow) lors des calculs
        r, g, b = cv2.split(rgb_img.astype(np.float32))
        
        avg_r = np.mean(r)
        avg_g = np.mean(g)
        avg_b = np.mean(b)
        
        # Sécurité pour éviter la division par zéro sur une image totalement noire
        if avg_r == 0 or avg_g == 0 or avg_b == 0:
            return rgb_img
            
        avg_gray = (avg_r + avg_g + avg_b) / 3.0
        
        # Application des facteurs d'échelle d'illumination et bornage [0, 255]
        r = np.clip(r * (avg_gray / avg_r), 0, 255)
        g = np.clip(g * (avg_gray / avg_g), 0, 255)
        b = np.clip(b * (avg_gray / avg_b), 0, 255)
        
        result = cv2.merge([r, g, b])
        return result.astype(np.uint8)

    def _create_dynamic_mask(self, hsv_img):
        """
        Génère un masque binaire dynamique excluant les reflets spéculaires et les ombres.
        """
        h, s, v = cv2.split(hsv_img)
        
        # Identification dynamique de la luminance maximale de l'image courante
        v_max = np.max(v)
        
        # 1. Masque des reflets : Très lumineux (> 90% du max local) ET peu saturé (< 30)
        # Les opérations bitwise d'OpenCV sont écrites en C, idéales pour la scalabilité
        highlight_mask = cv2.bitwise_and(
            (v > 0.9 * v_max).astype(np.uint8),
            (s < 30).astype(np.uint8)
        ) * 255
        
        # 2. Masque des ombres : Valeur de luminosité extrêmement faible (bruit capteur)
        shadow_mask = (v < 20).astype(np.uint8) * 255
        
        # 3. Masque final : On garde les pixels qui ne sont NI des reflets, NI des ombres
        bad_pixels = cv2.bitwise_or(highlight_mask, shadow_mask)
        valid_mask = cv2.bitwise_not(bad_pixels)
        
        return valid_mask

    def calculate_hsv_distance_pil(self, pil_img1, pil_img2):
        """
        Calcule la distance entre les histogrammes HSV de deux objets PIL Image.
        Robuste aux variations d'éclairage et aux réflectances des matériaux.
        """
        
        # 1. Conversion PIL -> NumPy (Format RGB)
        img1_np = np.array(pil_img1)
        img2_np = np.array(pil_img2)

        # 2. Correction de l'Illuminant (Color Constancy)
        img1_gw = self._apply_gray_world(img1_np)
        img2_gw = self._apply_gray_world(img2_np)

        # 3. Conversion vers l'espace HSV
        hsv1 = cv2.cvtColor(img1_gw, cv2.COLOR_RGB2HSV)
        hsv2 = cv2.cvtColor(img2_gw, cv2.COLOR_RGB2HSV)

        # 4. Création des masques dynamiques
        mask1 = self._create_dynamic_mask(hsv1)
        mask2 = self._create_dynamic_mask(hsv2)

        # 5. Calcul des histogrammes 2D (H et S) en appliquant les masques
        hist_size = [50, 60]
        ranges = [0, 180, 0, 256]

        hist1 = cv2.calcHist([hsv1], [0, 1], mask1, hist_size, ranges)
        hist2 = cv2.calcHist([hsv2], [0, 1], mask2, hist_size, ranges)
        
        # 6. Normalisation globale (indispensable puisque le masque modifie le nombre de pixels)
        cv2.normalize(hist1, hist1, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
        cv2.normalize(hist2, hist2, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)

        # 7. Calcul de la distance
        return cv2.compareHist(hist1, hist2, self._method)