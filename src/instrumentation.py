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


class Instrumented:
    """Accumulates what an operation cost: wall time, energy, carbon.

    The object half of the `timed` and `with_energy_consumption` decorators,
    which need `_time_it`, `_evaluate_energy_consumption` and the update
    methods. Mix it into anything whose cost is worth reporting; it says
    nothing about what that thing does.

    `time`, `energy` and `carbon` hold the last measured call, `total_*` the
    running sum -- read the right one, a report that prints `time` after a loop
    sees only the final iteration.
    """

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


    def update_time(self, time: int):
        self.time = time
        self.total_time += time

    def update_energy(self, energy: float):
        self.energy = energy
        self.total_energy += energy

    def update_carbon(self, carbon: float):
        self.carbon = carbon
        self.total_carbon += carbon

