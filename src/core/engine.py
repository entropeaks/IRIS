"""The retrieval engine: build an index over a gallery, then query it."""

from typing import List, Tuple
from tqdm import tqdm
from src.eval import Metric, Score
from src.instrumentation import Instrumented, timed, with_energy_consumption
from src.rerankers import Reranker
from ..feature_stores import FeatureStore
from src.distances.fusion import RRFBasedFusion

from transformers.image_utils import load_image
import torch
from torchvision.transforms import v2
from torch.utils.data import BatchSampler, DataLoader, Dataset, SequentialSampler
import numpy as np

from src.types import Feature, RetrievalChannel


def _reject_shuffling(dataloader: DataLoader) -> None:
    """Refuse a dataloader whose order will not match its dataset.

    The gallery index rows and the feature store entries are matched to
    dataset.images_paths positionally, so a shuffled loader silently pairs each
    image with another one's path.
    """
    sampler = dataloader.batch_sampler if dataloader.batch_sampler is not None else dataloader.sampler
    if isinstance(sampler, BatchSampler):
        sampler = sampler.sampler
    if not isinstance(sampler, SequentialSampler):
        raise ValueError(
            f"gallery dataloader must iterate in dataset order, got {type(sampler).__name__}; "
            f"build it without shuffle=True and without a custom sampler"
        )


    

class SearchEngine(Instrumented):
    """Retrieval over one or more feature channels, with optional reranking.

    Each `RetrievalChannel` pairs an extractor with an index and a distance
    kernel; their rankings are combined by `fusion_strategy`, and a `reranker`
    may reorder the top `top_k_candidates`. There is one pipeline shape here, so
    a search is configured entirely by the arguments passed in.
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
        self._gallery_prepared = False


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

        gallery_labels = torch.tensor(gallery_dataloader.dataset.labels)

        print("Computing queries features...")
        query_features, query_labels = self._compute_features(query_dataloader, update_index=False)
        query_labels = torch.tensor(query_labels)
        
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
        _reject_shuffling(gallery_dataloader)
        images_paths = self._extract_paths_from_dataloader(gallery_dataloader)
        features, _ = self._compute_features(gallery_dataloader, update_index=True)
        self._gallery_store.bulk_add(images_paths, features)
        self._gallery_prepared = True
        print(f"✅ Gallery prepared: {len(self._gallery_store)} images")

    
    def _extract_dataset_from_dataloader(self, dataloader: DataLoader) -> Dataset:
        return dataloader.dataset
    
    def _extract_paths_from_dataloader(self, dataloader: DataLoader) -> List[str]:
        """Extrait les chemins d'un dataloader (point de couplage isolé)."""
        return dataloader.dataset.images_paths
    
    
    def _compute_features(self, dataloader: DataLoader,
                          update_index: bool) -> Tuple[list[list[Feature]], list]:
        """Extract every channel in one pass, returning the labels seen alongside.

        The single pass matters: iterating once per channel lets a shuffling
        dataloader hand each channel a different order, and fusing those blocks
        would combine rows belonging to different images. Returning the labels
        from the same pass is what keeps them attached to their features, rather
        than being read back from the dataset in its own order.
        """
        features_blocks = [[] for _ in self._channels]
        labels = []

        for batch_imgs, batch_labels in tqdm(dataloader, desc="Computing features"):
            labels.extend(batch_labels)
            for block, batch_features in zip(features_blocks, self._extract(batch_imgs)):
                block.extend(batch_features)

        if update_index:
            for ch, features_block in zip(self._channels, features_blocks):
                if not ch.index.is_empty():
                    ch.index.update(features_block)
                else:
                    ch.index.build(features_block)

        return features_blocks, labels


    def is_gallery_prepared(self) -> bool:
        return self._gallery_prepared


    def _extract(self, images: list) -> list[list[Feature]]:
        """Run every channel over one batch of images, in channel order."""
        return [ch.extractor.get_features(images) for ch in self._channels]


    def _load(self, image_path: str) -> np.ndarray:
        return np.array(self._preprocessor(load_image(image_path)))


    def _rank_gallery(self, image: np.ndarray) -> np.ndarray:
        """Distances from one image to every gallery entry, reranked if configured."""
        if self._gallery_store is None or not len(self._gallery_store):
            raise ValueError("Gallery is empty. Call prepare_gallery() first.")

        dists = self._compute_distances(self._extract([image]))
        if self._reranker:
            dists = self._rerank([(image, None)], dists)
        return dists[0]
    
    
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

        fused_dists = self._fusion_strategy.fuse(feature_dists, [ch.weight for ch in self._channels])

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
    def inference(self, query_path: str) -> str:
        """Closest gallery entry to one image.

        Timed separately from find_nearest_neighbors so the cost of a single
        query stays measurable on its own.
        """
        return self._gallery_store[int(np.argmin(self._rank_gallery(self._load(query_path))))]


    @timed
    def find_nearest_neighbors(self, query_path: str, k: int) -> List[str]:
        """The k closest gallery entries to one image."""
        ranking = np.argsort(self._rank_gallery(self._load(query_path)))[:k]
        return self._gallery_store[ranking]


    def add_to_gallery(self, path: str) -> None:
        features = self._extract([self._load(path)])
        for ch, block in zip(self._channels, features):
            if ch.index.is_empty():
                ch.index.build(block)
            else:
                ch.index.update(block)
        # one feature per channel, the shape bulk_add stores
        self._gallery_store.add(path, [block[0] for block in features])
