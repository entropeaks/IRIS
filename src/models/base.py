from abc import ABC, abstractmethod
from ..eval import Metric, Score
from ..config import Config
from ..feature_stores import FeatureStore
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoProcessor
import torch
import time
import uuid
import random
from codecarbon import EmissionsTracker
from typing import Optional, List
from functools import wraps


def timed(func):
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        if self._time_it:
            t1 = time.time()
            result = func(self, *args, **kwargs)
            t2 = time.time()
            elapsed = t2-t1
            print(f"Elapsed time for {func.__name__} = {elapsed:.2f}s")
            self.update_time((elapsed))
        else:
            result = func(self, *args, **kwargs)
        
        return result

    return wrapper


def with_energy_consumption(func):
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        if self._evaluate_energy_consumption:
            try:
                tracker = EmissionsTracker(log_level="critical", save_to_file=False)
                tracker.start_task(func.__name__)
                result = func(self, *args, **kwargs)
                tracker_results = tracker.stop_task(func.__name__)
                energy = tracker_results.energy_consumed
                carbon_footprint = tracker_results.emissions
                print(f"Energy consumption for {func.__name__} = {energy:.6f} kWh")
                print(f"Carbon footprint for {func.__name__} = {carbon_footprint:.6f} g.eq.CO2")
                self.update_carbon(carbon_footprint)
                self.update_energy(energy)
            except Exception as err:
                print(f"Energy consumption unreadable: {err}")
                result = func(self, *args, **kwargs)
            finally:
                _ = tracker.stop()
        else:
            result = func(self, *args, **kwargs)
        
        return result

    return wrapper


def unique_readable_name():
    adjectives = ["brave", "curious", "eager", "silent", "clever", "wild", "cosmic", "steady"]
    nouns = ["otter", "phoenix", "comet", "falcon", "nebula", "quark", "mamba", "lynx"]
    uid = uuid.uuid4().hex[:6]
    return f"{random.choice(adjectives)}-{random.choice(nouns)}-{uid}"


class BaseModel(ABC):

    def __init__(self,
                 time_it: bool=True,
                 evaluate_energy_consumption: bool=True,
                 ):
        self._time_it = time_it
        self._evaluate_energy_consumption = evaluate_energy_consumption
        self.time = None
        self.energy = None
        self.carbon = None
        self.total_time = 0
        self.total_energy = 0
        self.total_carbon = 0

        self._gallery_prepared = False
        self.gallery_store: FeatureStore = None

    def update_time(self, time: int):
        self.time = time
        self.total_time += time

    def update_energy(self, energy: float):
        self.energy = energy
        self.total_energy += energy

    def update_carbon(self, carbon: float):
        self.carbon = carbon
        self.total_carbon += carbon

    @abstractmethod
    def evaluate(self, metric: Metric) -> Score:
        pass

    @abstractmethod
    def inference(self, ref_path: str):
        pass

    @abstractmethod
    def find_nearest_neighbors(self, query_path: str, k: int) -> str:
        pass

    def prepare_gallery(self, gallery_dataloader: DataLoader) -> None:
        """
        Pré-calcule et met en cache les features/embeddings de la gallery.
        
        Cette méthode est optionnelle mais FORTEMENT RECOMMANDÉE pour les performances.
        Elle sera appelée automatiquement par evaluate() si nécessaire.
        
        Par défaut, ne fait rien (pour les modèles qui ne supportent pas le cache).
        Les sous-classes peuvent la surcharger pour implémenter le caching.
        
        Args:
            gallery_dataloader: DataLoader de la gallery
        """
        self._gallery_prepared = True

    def is_gallery_prepared(self) -> bool:
        return self._gallery_prepared


class DeepModel(BaseModel):

    def __init__(self,
                 config: Config,
                 time_it: bool=True,
                 evaluate_energy_consumption: bool=True
                 ):
        super().__init__(time_it, evaluate_energy_consumption)
        self.config = config
        self.name = unique_readable_name()
    
    @abstractmethod
    @with_energy_consumption
    @timed
    def fit(self, dataloader: DataLoader) -> None:
        pass
    
    @abstractmethod
    @with_energy_consumption
    @timed
    def fit_and_evaluate(self, epochs: int, metric: Metric):
        pass
    
    @abstractmethod
    def save(self):
        pass


class CVModel(BaseModel):
    def __init__(self):
        pass