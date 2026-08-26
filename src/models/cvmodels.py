#This file contains all the computer vision based models implementations

from typing import List
from tqdm import tqdm
from src.eval import Metric, Score
from src.models.base import BaseModel, timed, with_energy_consumption
from src.models.rerankers import Reranker
from ..feature_stores import FeatureStore
from src.distances.fusion import RRFBasedFusion

from transformers.image_utils import load_image
import torch
from torchvision.transforms import v2
from torch.utils.data import DataLoader, Dataset
import numpy as np

from src.types import Feature, RetrievalChannel


    

class FeatureBasedEstimator(BaseModel):
    """
    Estimateur basé sur des features extraites.
    Utilise un FeatureStore pour le cache et un DistanceCalculator pour les distances.
    """
    
    def __init__(
        self,
        preprocessor: v2.Compose,
        channels: List[RetrievalChannel],
        gallery_store: FeatureStore,
        fusion_strategy: RRFBasedFusion,
        reranker: Reranker=None,
        top_k_candidates: int=50,
        time_it: bool=True,
        evaluate_energy_consumption: bool=True
    ):
        """
        Args:
            feature_extractors: Liste des extracteurs de features
            weights: Poids de chaque feature (doit sommer à 1.0)
            time_it: Si True, mesure le temps d'exécution
            evaluate_energy_consumption: Si True, mesure la consommation énergétique
        """

        super().__init__(time_it, evaluate_energy_consumption)

        self._preprocessor = preprocessor
        self._channels = channels
        self._gallery_store = gallery_store
        self._fusion_strategy = fusion_strategy
        self._reranker = reranker
        self._top_k_candidates = top_k_candidates
        self._gallery_dataset: Dataset = None


    def fit(self, train_dataloader: DataLoader):
        for ch in self._channels:
            if ch.is_trainable:
                ch.extractor.fit(train_dataloader)
    

    @with_energy_consumption
    @timed
    def evaluate(
        self,
        gallery_dataloader: DataLoader,
        query_dataloader: DataLoader,
        metric: Metric
    ) -> Score:
        """
        Évalue le modèle sur gallery et query.
        
        Args:
            gallery_dataloader: DataLoader de la gallery
            query_dataloader: DataLoader des queries
            metric: Métrique d'évaluation
            
        Returns:
            Score d'évaluation
        """

        if not self._gallery_prepared:
            self.prepare_gallery(gallery_dataloader)

        query_labels = torch.tensor(query_dataloader.dataset.labels)
        gallery_labels = torch.tensor(gallery_dataloader.dataset.labels)

        print("Computing queries features...")
        query_features = self._compute_features(query_dataloader, update_index=False)
        
        distances = self._compute_distances(query_features)
        scores = metric.compute(torch.from_numpy(distances),
                                query_labels,
                                gallery_labels)
        
        if self._reranker:
            print(f"Scores before reranking:\n{scores}")
            query_dataset = self._extract_dataset_from_dataloader(query_dataloader)
            new_distances = self._rerank(query_dataset, distances)
            scores = metric.compute(
                torch.from_numpy(new_distances),
                query_labels,
                gallery_labels
            )

            print(f"Scores after reranking:\n{scores}")
        
        return scores
    
    
    def prepare_gallery(self, gallery_dataloader: DataLoader):
        print("🔨 Preparing gallery (computing features)...")
        self._gallery_dataset = self._extract_dataset_from_dataloader(gallery_dataloader)
        images_paths = self._extract_paths_from_dataloader(gallery_dataloader)
        features = self._compute_features(gallery_dataloader, update_index=True)
        self._gallery_store.bulk_add(images_paths, features)
        self._gallery_prepared = True
        print(f"✅ Gallery prepared: {len(self._gallery_store)} images")

    
    def _extract_dataset_from_dataloader(self, dataloader: DataLoader) -> Dataset:
        return dataloader.dataset
    
    def _extract_paths_from_dataloader(self, dataloader: DataLoader) -> List[str]:
        """Extrait les chemins d'un dataloader (point de couplage isolé)."""
        return dataloader.dataset.images_paths
    
    
    def _compute_features(self, dataloader: DataLoader, update_index: bool) -> list[list[Feature]]:
        features_blocks = []
        for ch in tqdm(self._channels, desc="Computing features blocks"):
            features_block = self._compute_channel_features(ch, dataloader)
            features_blocks.append(features_block)
            if update_index:
                if not ch.index.is_empty():
                    ch.index.update(features_block)
                else:
                    ch.index.build(features_block)

        return features_blocks
    
    
    def _compute_channel_features(self, channel: RetrievalChannel, dataloader: DataLoader) -> list[Feature]:
        features = []
        for batch_imgs, _ in dataloader:
            features.extend(channel.extractor.get_features(batch_imgs))

        return features
    
    
    def _compute_distances(self, query_features: list[list[Feature]]):
        feature_dists = []
        
        for i, ch in enumerate(self._channels):
            dists = ch.index.distances(query_features[i])

            print(f"Feature {i}: min={dists.min():.4f}, max={dists.max():.4f}, "
            f"std={dists.std():.4f}, mean={dists.mean():.4f}")

            normalized_dists = self._fusion_strategy.normalize_distances(dists)

            print(f"Feature {i} after norm: min={normalized_dists.min():.4f}, max={normalized_dists.max():.4f}, "
            f"std={normalized_dists.std():.4f}, mean={normalized_dists.mean():.4f}")
            
            feature_dists.append(normalized_dists)

        fused_dists = self._fusion_strategy.fuse(feature_dists)

        return fused_dists
    
    
    def _rerank(self, query_imgs: Dataset, dists: list[list[float|int]]):
        candidates_indices = np.argsort(dists, axis=1)[:, :self._top_k_candidates]
        candidates_dists = self._reranker.score(query_imgs, self._gallery_dataset, candidates_indices)

        nQ = dists.shape[0]
        row_indices = np.arange(nQ)[:, None]
        original_dists_for_candidates = dists[row_indices, candidates_indices]

        # the reranker is fused as one more channel over the shortlist rather than
        # added to the retrieval distance: the two live on unrelated scales, so the
        # sum gave the reranker a weight that drifted with the gallery size
        final_candidates_scores = self._fusion_strategy.fuse([
            self._fusion_strategy.normalize_distances(original_dists_for_candidates),
            self._fusion_strategy.normalize_distances(candidates_dists),
        ])

        new_dists = np.full(dists.shape, np.inf)
        for i in range(len(candidates_dists)):
            new_dists[i, candidates_indices[i]] = final_candidates_scores[i]

        return new_dists
    
    
    @timed
    def inference(self, query_path: str) -> List:

        if self._gallery_store is None:
            raise ValueError(
                "Gallery store not initialized. Call prepare_gallery() first."
            )
        
        img = load_image(query_path)
        img = self._preprocessor(img)
        img = np.array(img)

        query_features = self._compute_features([[[img], []]], update_index=False)

        dists = self._compute_distances(query_features)
        if self._reranker:
            dists = self._rerank([[img], []], dists)

        idx = np.argmin(dists)

        return self._gallery_store[idx]
    
    @timed
    def find_nearest_neighbors(self, query_path, k):

        if self._gallery_store is None:
            raise ValueError(
                "Gallery store not initialized. Call prepare_gallery() first."
            )
        
        img = load_image(query_path)
        img = self._preprocessor(img)
        img = np.array(img)

        query_features = self._compute_features([[[img], []]], update_index=False)

        dists = self._compute_distances(query_features)
        if self._reranker:
            dists = self._rerank([[img], []], dists)

        idx = np.argsort(dists)[:, :k][0]


        return self._gallery_store[idx]
    

    def add_to_gallery(self, path: str) -> None:
        img = load_image(path)
        img = self._preprocessor(img)
        img = np.array(img)

        img_features = self._compute_features([[[img], []]], update_index=True)
        self._gallery_store.add(path, img_features[0])

    """ def _fit_extractors(self, feature_list: List):
        for feature_index, fe in enumerate(self.feature_extractors):
            fe.fit([entry_features[feature_index] for entry_features in feature_list]) """