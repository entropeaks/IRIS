import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoProcessor
from .base import DeepModel, timed, with_energy_consumption
from ..feature_stores import InMemoryStore
from ..distances import VectorBasedDistance
from ..eval import Metric
from ..config import Config
from ..utils import set_device
from torch.utils.data import DataLoader
import torch
import numpy as np
from tqdm import tqdm
from typing import List
from PIL import Image
from wandb import Run

class MockRun:
    def __getattr__(self, name):
        # Retourne une fonction qui ne fait rien pour n'importe quel nom de méthode
        return lambda *args, **kwargs: None


class SiameseDino(DeepModel, nn.Module):
    def __init__(self,
                 backbone: AutoModel,
                 processor: AutoProcessor,
                 config: Config,
                 run: Run=None,
                 time_it: bool=True,
                 evaluate_energy_consumption: bool=True
                 ):
        DeepModel.__init__(self, config, time_it, evaluate_energy_consumption)
        nn.Module.__init__(self)
        self.backbone = backbone
        self.optimizer = None
        self.processor = processor
        embedding_dim = backbone.config.hidden_size
        self.projection_head = nn.Sequential(
            nn.Linear(embedding_dim, self.config.model.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.config.model.dropout),
            nn.Linear(self.config.model.hidden_dim, self.config.model.output_dim)
            ) if self.config.model.hidden_dim > 0 else nn.Sequential(nn.Linear(embedding_dim, self.config.model.output_dim), nn.Dropout(self.config.model.dropout))
        self.loss = nn.TripletMarginLoss(margin=self.config.train.margin, p=2)
        self.device = set_device(config.base.device)
        self.to(self.device)

        self.distance_strategy = VectorBasedDistance()
        self.gallery_labels = None
        self.run = run or MockRun()


    def set_optimizer(self, optimizer: torch.optim.Optimizer):
        self.optimizer = optimizer

    def set_run(self, run: Run):
        self.run = run

    def forward(self, **inputs):
        outputs = self.backbone(**inputs)
        x = outputs.pooler_output if hasattr(outputs, 'pooler_output') else outputs.last_hidden_state.mean(dim=1)
        x = self.projection_head(x)
        if self.config.model.normalize:
            x = F.normalize(x, p=2, dim=1)
        return x
    

    def to(self, device):
        self.backbone.to(device)
        self.projection_head.to(device)
        self.device = device
        return self
    

    #@with_energy_consumption
    @timed
    def inference(self, ref_path: str) -> torch.Tensor:
        self.eval()
        img = Image.open(ref_path)
        
        with torch.no_grad():
            inputs = self.processor(images=[img], return_tensors="pt").to(self.device)
            embedding = self(**inputs)

        return embedding.cpu()
    
    @timed
    def find_nearest_neighbors(self, query_path: str, k: int):
        if not self.is_gallery_prepared():
            raise ValueError("Gallery not prepared. Call prepare_gallery() first.")
        
        query_embedding = self.inference(query_path)
        knn_paths = self.gallery_store.search(query_embedding, k)
        return knn_paths


    def fit(self, dataloader: DataLoader) -> None:
        pass
    
    #@with_energy_consumption
    @timed
    def fit_and_evaluate(self,
                         train_dataloader: DataLoader,
                         gallery_dataloader: DataLoader,
                         query_dataloader: DataLoader,
                         metric: Metric):
        best_score = 0.0
        best_metrics = {}
        for epoch in tqdm(range(self.config.train.epochs)):
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
            inputs = self.processor(images=images, return_tensors="pt").to(self.device)
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
                cond2 = dists[anchor_idx, neg_indices] < (pos_dist + self.config.train.margin)
                
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

        dists = self.distance_strategy.compute_distances_batch(
            query_embeddings,
            gallery_embeddings
        )

        scores = metric.compute(dists, query_labels, gallery_labels)

        return scores

    @timed
    @torch.no_grad()
    def evaluate(self, gallery_dataloader: DataLoader, query_dataloader: DataLoader, metric: Metric) -> dict:
        """
        Compute Recall@k in pure PyTorch.
        
        Args:
            gallery_loader: DataLoader with (image, label) for gallery
            query_loader: DataLoader with (image, label) for queries
            metric: Metric to evaluate the model against
            device: str
        
        Returns:
            recall_at_k: list
        """
        self.eval()

        if not self.is_gallery_prepared():
            self.prepare_gallery(gallery_dataloader)

        query_embeddings, query_labels = self._compute_embeddings(query_dataloader)

        dists = self.distance_strategy.compute_distances_batch(
            query_embeddings,
            torch.stack(self.gallery_store.get_feature_gallery())
        )

        scores = metric.compute(dists, query_labels, self.gallery_labels)

        return scores
    

    def prepare_gallery(self, gallery_dataloader: DataLoader) -> None:
        self.gallery_store = InMemoryStore(self.distance_strategy)
        print("🔨 Preparing gallery (computing embeddings)...")
        gallery_embeddings, gallery_labels = self._compute_embeddings(gallery_dataloader)
        gallery_paths = gallery_dataloader.dataset.images_paths
        self.gallery_store.bulk_add(zip(gallery_paths, gallery_embeddings))
        self.gallery_labels = gallery_labels
        self._gallery_prepared = True
        print(f"✅ Gallery prepared: {len(self.gallery_store)} images")
    
    
    def _compute_embeddings(self, dataloader: DataLoader):
        self.eval()

        embeddings, all_labels = [], []

        with torch.no_grad():
            for images, labels in dataloader:
                inputs = self.processor(images=images, return_tensors="pt").to(self.device)
                emb = self(**inputs)
                if not self.config.model.normalize:
                    emb = F.normalize(emb, p=2, dim=1)
                embeddings.append(emb.cpu())
                all_labels.append(labels)
        
        embeddings = torch.cat(embeddings, dim=0)
        all_labels = torch.cat(all_labels, dim=0)

        return embeddings, all_labels
    

    def _model_improvement(self, metrics: dict, best_score: float) -> bool:
        return np.mean([score for score in metrics.values()]) > best_score


    def save(self):
        torch.save(self.state_dict(), f"{self.config.base.model_checkpoints_path}/{self.name}.pth")
            

class PrototypicalNetwork(DeepModel, nn.Module):
    def __init__(self, backbone, output_dim, normalize: bool=True, dropout: float=0.0):
        super().__init__()
        pass