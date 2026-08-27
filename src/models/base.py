from abc import ABC, abstractmethod
from ..eval import Metric, Score
from torch.utils.data import DataLoader
import time
from codecarbon import EmissionsTracker
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
    """Measure the energy a method draws, without letting the meter affect it.

    Failures in codecarbon are reported and the call proceeds untracked; failures
    in the wrapped method propagate. Only the measurement is best-effort.
    """
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        if not self._evaluate_energy_consumption:
            return func(self, *args, **kwargs)

        try:
            tracker = EmissionsTracker(log_level="critical", save_to_file=False)
            tracker.start_task(func.__name__)
        except Exception as err:
            print(f"Energy tracking unavailable, running untracked: {err}")
            return func(self, *args, **kwargs)

        try:
            result = func(self, *args, **kwargs)
        finally:
            try:
                measurement = tracker.stop_task(func.__name__)
                tracker.stop()
            except Exception as err:
                print(f"Energy consumption unreadable: {err}")
                measurement = None

        if measurement is not None:
            print(f"Energy consumption for {func.__name__} = {measurement.energy_consumed:.6f} kWh")
            print(f"Carbon footprint for {func.__name__} = {measurement.emissions:.6f} g.eq.CO2")
            self.update_carbon(measurement.emissions)
            self.update_energy(measurement.energy_consumed)

        return result

    return wrapper


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
