#This file contains all the computer vision based models implementations
import cv2
from typing import List, Tuple, Optional
from tqdm import tqdm
from src.eval import Metric, Recall, Score
from src.models.base import BaseModel, DeepModel, timed, with_energy_consumption
from ..feature_stores import InMemoryStore
from ..distances import FeatureBasedDistance
from .feature_extractors import FeatureExtractor
import numpy as np
import torch
from torch.utils.data import DataLoader


class FeatureBasedEstimator(BaseModel):
    """
    Estimateur basé sur des features extraites.
    Utilise un FeatureStore pour le cache et un DistanceCalculator pour les distances.
    """
    
    def __init__(
        self,
        feature_extractors: List['FeatureExtractor'],
        weights: List[float],
        time_it: bool = True,
        evaluate_energy_consumption: bool = True
    ):
        """
        Args:
            feature_extractors: Liste des extracteurs de features
            weights: Poids de chaque feature (doit sommer à 1.0)
            time_it: Si True, mesure le temps d'exécution
            evaluate_energy_consumption: Si True, mesure la consommation énergétique
        """

        super().__init__(time_it, evaluate_energy_consumption)

        # Validation des paramètres
        if len(feature_extractors) != len(weights):
            raise ValueError(
                f"Mismatch: {len(feature_extractors)} extractors but {len(weights)} weights"
            )
        
        weights_array = np.array(weights, dtype=np.float32)
        if not np.isclose(weights_array.sum(), 1.0):
            raise ValueError(f"Weights must sum to 1.0, got {weights_array.sum()}")
        
        self.feature_extractors = feature_extractors
        self.weights = weights_array
        
        self.distance_strategy = FeatureBasedDistance(feature_extractors, weights_array)
    
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

        query_paths = self._extract_paths_from_dataloader(query_dataloader)
        print("Computing queries features...")
        query_features = self._compute_features_from_paths(query_paths)
        
        distances = self.distance_strategy.compute_distances_batch(
            query_features, self.gallery_store.get_feature_gallery()
        )
        
        query_labels = torch.tensor(query_dataloader.dataset.labels)
        gallery_labels = torch.tensor(gallery_dataloader.dataset.labels)
        
        return metric.compute(
            torch.from_numpy(distances),
            query_labels,
            gallery_labels
        )
    
    
    def prepare_gallery(self, gallery_dataloader: DataLoader):
        self.gallery_store = InMemoryStore(self.distance_strategy)
        print("🔨 Preparing gallery (computing features)...")
        images_paths = self._extract_paths_from_dataloader(gallery_dataloader)
        features = self._compute_features_from_paths(images_paths)
        print(features)
        self.gallery_store.bulk_add(zip(images_paths, features))
        self._gallery_prepared = True
        self._fit_extractors(features)
        print(f"✅ Gallery prepared: {len(self.gallery_store)} images")
    
    @timed
    def inference(self, query_path: str) -> List:
        query_features = [fe.get_features(query_path) for fe in self.feature_extractors]
        return query_features

    @with_energy_consumption
    @timed
    def find_nearest_neighbors(self, query_path: str, k: int) -> str:
        """
        Trouve le voisin le plus proche dans la gallery.
        
        Args:
            query_path: Chemin de l'image query
            
        Returns:
            Chemin de l'image la plus proche
            
        Raises:
            ValueError: Si la gallery n'a pas été initialisée
        """
        if self.gallery_store is None:
            raise ValueError(
                "Gallery store not initialized. Call prepare_gallery() first."
            )
        
        query_features = self.inference(query_path)
        
        paths = self.gallery_store.search(query_features, k)
        
        return paths
    
    def _extract_paths_from_dataloader(self, dataloader: DataLoader) -> List[str]:
        """Extrait les chemins d'un dataloader (point de couplage isolé)."""
        return dataloader.dataset.images_paths
    
    def _compute_features_from_paths(self, paths: List[str]) -> List[List[np.ndarray]]:
        """Calcule les features pour une liste de chemins (sans cache)."""
        features = []
        for path in tqdm(paths, desc="Computing features"):
            img_features = [fe.get_features(path) for fe in self.feature_extractors]
            features.append(img_features)
        return features
    

    def _fit_extractors(self, feature_list: List):
        for feature_index, fe in enumerate(self.feature_extractors):
            fe.fit([entry_features[feature_index] for entry_features in feature_list])