from abc import ABC, abstractmethod

from src.models.feature_extractors import FeatureExtractor
from src.models.rerankers import Reranker
from src.feature_stores import FeatureStore

class Recipe(ABC):

    @abstractmethod
    def apply():
        pass

class PipelineRecipe(Recipe):

    def __init__(self,
                 retriever: FeatureExtractor,
                 reranker: Reranker,
                 top_k_retriever: int=50):
        
        #TODO add the possibility to fuse multiple retrievers' outputs

        self._retriever = retriever
        self._reranker = reranker
        self._top_k_retriever = top_k_retriever

    
    def apply(self, query_imgs: list, feature_store: FeatureStore):

        candidates = None

        final_results = None


class LateFusionModel(Recipe):

    def __init__(self,
                 retrievers: list[FeatureExtractor],
                 fusion_strategy):
        self._retrievers = retrievers
        self._fusion_strategy = fusion_strategy

    def apply(self, query_imgs: list, feature_store: FeatureStore):
        pass