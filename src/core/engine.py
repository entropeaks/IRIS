from typing import Optional

from src.models.feature_extractors import FeatureExtractor
from src.models.rerankers import Reranker
from src.feature_stores import FeatureStore

class SearchEngine():

    def __init__(self, recipe, feature_store: FeatureStore):
        self._recipe = recipe
        self._feature_store = feature_store

    def set_recipe(self, new_recipe):
        self._recipe = new_recipe

    def search(self, img_path: str) -> str:
        raise NotImplementedError()