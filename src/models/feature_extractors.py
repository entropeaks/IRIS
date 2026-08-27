
from pathlib import Path
from typing import TYPE_CHECKING, List, Tuple
import uuid
import numpy as np
import cv2
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.config import Config
from src.types import FeatureExtractor
from src.utils import set_device
from src.eval import Metric

# doctr, transformers and wandb cost roughly 2.7s, 1.0s and 0.5s to import and
# are each needed by a single class here. Importing them where they are used
# keeps `import src.models.rerankers` -- which only needs OpenCV -- from paying
# for all three.
if TYPE_CHECKING:
    from wandb import Run

    
class DocTRTextExtractor(FeatureExtractor):

    def __init__(self):
        from doctr.models import ocr_predictor

        super().__init__()
        # half precision is a CUDA-only win here; several ops fall back or fail
        # on cpu and mps, so keep full precision off CUDA
        device = set_device("auto")
        self.ocr = ocr_predictor(
            det_arch="db_mobilenet_v3_large",
            reco_arch="crnn_mobilenet_v3_small",
            pretrained=True
        ).to(device)
        if device.type == "cuda":
            self.ocr = self.ocr.half()
    
    def get_features(self, imgs_arrays_rgb: list[np.ndarray]) -> list[list[str]]:
        result = self.ocr(imgs_arrays_rgb)
        
        return [
            [
                word.value.lower()
                for block in page.blocks
                for line in block.lines
                for word in line.words
            ]
            for page in result.pages
        ]
    
    def fit(self):
        pass


class OrbFeatureExtractor(FeatureExtractor):

    def __init__(self, n_features: int=500):
        self.orb = cv2.ORB_create(nfeatures=n_features)
        self.kmeans = None

    def get_features(self, imgs_arrays_rgb: list[np.ndarray]):
        all_descriptors = []
        for img in imgs_arrays_rgb:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            _, descriptors = self.orb.detectAndCompute(img, None)
            all_descriptors.append(descriptors)

        return all_descriptors
    
    def fit(self):
        pass
    

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
    
    def fit(self):
        pass


class HSVExtractor(FeatureExtractor):

    def __init__(self, hist_size: list=[50, 60]):
        self.trainable = False
        self._hist_size = hist_size


    def get_features(self, imgs_arrays_rgb: list[np.ndarray]):
        features = []
        for img_rgb in imgs_arrays_rgb:
            img_gw = self._apply_gray_world(img_rgb)
            img_hsv = cv2.cvtColor(img_gw, cv2.COLOR_RGB2HSV)
            mask = self._create_dynamic_mask(img_hsv)

            hist_size = self._hist_size
            ranges = [0, 180, 0, 256]
            
            hist = cv2.calcHist([img_hsv], [0, 1], mask, hist_size, ranges)
            cv2.normalize(hist, hist, alpha=1, beta=0, norm_type=cv2.NORM_L1)
            features.append(hist)

        return features


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
    

    def fit(self):
        pass
    


class MockRun:
    def __getattr__(self, name):
        # Retourne une fonction qui ne fait rien pour n'importe quel nom de méthode
        return lambda *args, **kwargs: None


class SiameseDino(FeatureExtractor, nn.Module):
    def __init__(self, config: Config):

        nn.Module.__init__(self)

        self._config = config
        from transformers import AutoModel, AutoImageProcessor

        self._backbone = AutoModel.from_pretrained(self._config.model.backbone_name)
        self.optimizer = None
        resize = {"height": self._config.train.resize.height, "width": self._config.train.resize.width}
        self._processor = AutoImageProcessor.from_pretrained(self._config.model.backbone_name, size=resize)
        
        #n_prefix is num_registers + 1 to take all patch tokens without CLS and register tokens
        self._n_prefix = self._backbone.config.num_register_tokens + 1
        
        embedding_dim = self._backbone.config.hidden_size
        self.projection_head = nn.Sequential(
            nn.Linear(embedding_dim, self._config.model.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self._config.model.dropout),
            nn.Linear(self._config.model.hidden_dim, self._config.model.output_dim)
            ) if self._config.model.hidden_dim > 0 else nn.Sequential(nn.Linear(embedding_dim, self._config.model.output_dim), nn.Dropout(self._config.model.dropout))
        self.loss = nn.TripletMarginLoss(margin=self._config.train.margin, p=2)
        self.device = set_device(config.base.device)
        self.to(self.device)

        self.gallery_labels = None
        self._name = f"siamese-{uuid.uuid4().hex[:6]}"
        import wandb

        self.run = wandb.init(project=config.base.wandb_project_name, entity=config.base.wandb_entity, config=self._config) or MockRun()


    def set_optimizer(self, optimizer: torch.optim.Optimizer):
        self.optimizer = optimizer

    def set_run(self, run: "Run"):
        self.run = run

    def gem_pooling(self, patch_tokens, p=3):
        # patch_tokens: (1, N_patches, hidden_size)
        return patch_tokens.clamp(min=1e-6).pow(p).mean(dim=1).pow(1/p)

    def forward(self, **inputs):
        outputs = self._backbone(**inputs)
        x = outputs.last_hidden_state[:, self._n_prefix:, :] 
        x = self.gem_pooling(x)
        x = self.projection_head(x)
        if self._config.model.normalize:
            x = F.normalize(x, p=2, dim=1)
        return x
    

    def to(self, device):
        self._backbone.to(device)
        self.projection_head.to(device)
        self.device = device
        return self
    
    
    @torch.no_grad
    def get_features(self, imgs_arrays_rgb: list[np.ndarray]):
        inputs = self._processor(images=imgs_arrays_rgb, return_tensors="pt").to(self.device)
        embeddings = self(**inputs)

        return embeddings.cpu().numpy()


    def fit(self, dataloader: DataLoader) -> None:
        self.train()
        print("--------------- Training Siamese model ---------------")
        for epoch in tqdm(range(self._config.train.epochs)):
            train_metrics = self._fit_one_epoch(dataloader)
            print(f"Epoch {epoch+1}: {train_metrics}")


    def fit_and_evaluate(self,
                         train_dataloader: DataLoader,
                         gallery_dataloader: DataLoader,
                         query_dataloader: DataLoader,
                         metric: Metric):
        best_score = 0.0
        best_metrics = {}
        for epoch in tqdm(range(self._config.train.epochs)):
            train_metrics = self._fit_one_epoch(train_dataloader)
            metrics = self._evaluate_new_iteration(gallery_dataloader, query_dataloader, metric)
            
            train_metrics.update(metrics)
            self.run.log(train_metrics)

            if self._model_improvement(metrics, best_score):
                best_metrics = metrics
                best_score = np.mean([score for score in best_metrics.values()])
                self.save()

        return best_metrics


    def _fit_one_epoch(self, train_dataloader: DataLoader) -> dict:
        self.train()
        cumulative_loss = 0.0
        cumulative_pos_dist = 0.0
        cumulative_neg_dist = 0.0
        cumulative_triplets_count = 0
        for images, labels in train_dataloader:
            inputs = self._processor(images=images, return_tensors="pt").to(self.device)
            embeddings = self(**inputs)
            triplets = self._mine_semi_hard_triplets_cdist(embeddings, labels)
            if not triplets:
                continue
            cumulative_triplets_count += len(triplets)
            anchor_indices, positive_indices, negative_indices = zip(*triplets)
            anchor_embeddings = embeddings[list(anchor_indices)]
            positive_embeddings = embeddings[list(positive_indices)]
            negative_embeddings = embeddings[list(negative_indices)]
            anchor_embeddings = anchor_embeddings.to(self.device)
            positive_embeddings = positive_embeddings.to(self.device)
            negative_embeddings = negative_embeddings.to(self.device)
            self.optimizer.zero_grad()
            triplet_loss = self.loss(anchor_embeddings, positive_embeddings, negative_embeddings)
            triplet_loss.backward()
            self.optimizer.step()
            cumulative_pos_dist += F.pairwise_distance(anchor_embeddings, positive_embeddings, p=2).mean().item()
            cumulative_neg_dist += F.pairwise_distance(anchor_embeddings, negative_embeddings, p=2).mean().item()
            cumulative_loss += triplet_loss.detach().cpu().item()

        cumulative_loss /= len(train_dataloader)
        cumulative_pos_dist /= len(train_dataloader)
        cumulative_neg_dist /= len(train_dataloader)

        return {"loss": cumulative_loss,
                "positive_dist": cumulative_pos_dist,
                "negative_dist": cumulative_neg_dist,
                "triplets_mined": cumulative_triplets_count}

    @torch.no_grad
    def _mine_semi_hard_triplets_cdist(self, embeddings: torch.Tensor, labels: np.ndarray) -> List[tuple]:
        """
        Vectorized semi-hard negative mining using torch.cdist.
        For each anchor, finds a random positive and all semi-hard negatives.
        A semi-hard negative `n` satisfies: d(a, p) < d(a, n) < d(a, p) + margin
        
        embeddings: torch.Tensor of shape (N, D)
        labels: list or np.array of length N
        margin: float, the margin used in the TripletLoss
        
        Returns: list of (anchor_idx, positive_idx, semi_hard_negative_idx)
        """
        if isinstance(embeddings, list):
            embeddings = torch.stack(embeddings)
        
        if isinstance(labels, torch.Tensor):
            labels = labels.cpu().numpy()
        elif isinstance(labels, Tuple) or isinstance(labels, list):
            labels = np.array(labels)

        n = embeddings.shape[0]
        # Calcule la matrice des distances au carré pour la stabilité, ou p=2 pour euclidienne
        dists = torch.cdist(embeddings, embeddings, p=2)
        
        triplets = []
        for anchor_idx in range(n):
            anchor_label = labels[anchor_idx]
            
            # Masques pour positifs et négatifs
            pos_mask = (labels == anchor_label) & (np.arange(n) != anchor_idx)
            pos_indices = np.where(pos_mask)[0]
            
            neg_mask = (labels != anchor_label)
            neg_indices = np.where(neg_mask)[0]
            
            if len(pos_indices) == 0 or len(neg_indices) == 0:
                continue
                
            # Itérer sur tous les positifs possibles pour cet ancre
            for positive_idx in pos_indices:
                pos_dist = dists[anchor_idx, positive_idx]

                # Condition 1: d(a, n) > d(a, p)
                cond1 = dists[anchor_idx, neg_indices] > pos_dist
                # Condition 2: d(a, n) < d(a, p) + margin
                cond2 = dists[anchor_idx, neg_indices] < (pos_dist + self._config.train.margin)
                
                semi_hard_neg_mask = cond1 & cond2
                
                semi_hard_indices = neg_indices[semi_hard_neg_mask.cpu().numpy()]
                
                for semi_hard_neg_idx in semi_hard_indices:
                    triplets.append((anchor_idx, positive_idx, semi_hard_neg_idx))
                    
        return triplets
    

    @torch.no_grad()
    def _evaluate_new_iteration(self, gallery_dataloader: DataLoader, query_dataloader, metric: Metric) -> dict:
        self.eval()
        gallery_embeddings, gallery_labels = self._compute_embeddings(gallery_dataloader)
        query_embeddings, query_labels = self._compute_embeddings(query_dataloader)

        dists = self.compute_distances(
            query_embeddings,
            gallery_embeddings
        )

        scores = metric.compute(dists, query_labels, gallery_labels)

        return scores
    
    
    def compute_distances(
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
    

    def _compute_embeddings(self, dataloader: DataLoader):
        self.eval()

        embeddings, all_labels = [], []

        with torch.no_grad():
            for images, labels in dataloader:
                inputs = self._processor(images=images, return_tensors="pt").to(self.device)
                emb = self(**inputs)
                if not self._config.model.normalize:
                    emb = F.normalize(emb, p=2, dim=1)
                embeddings.append(emb.cpu())
                all_labels.append(labels)
        
        embeddings = torch.cat(embeddings, dim=0)
        all_labels = torch.cat(all_labels, dim=0)

        return embeddings, all_labels
    

    def _model_improvement(self, metrics: dict, best_score: float) -> bool:
        return np.mean([score for score in metrics.values()]) > best_score


    def save(self, name: str=None):
        """Write the state dict under the configured checkpoint directory."""
        directory = Path(self._config.base.model_checkpoints_path)
        directory.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), directory / f"{name or self._name}.pth")
            
