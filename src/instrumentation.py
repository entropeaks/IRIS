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
            self.record_time(func.__name__, elapsed)
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
            self.record_energy(func.__name__, measurement.energy_consumed,
                               measurement.emissions)

        return result

    return wrapper


class Instrumented:
    """Records what each stage cost: wall time, energy, carbon.

    The object half of the `timed` and `with_energy_consumption` decorators,
    which need `_time_it` and `_evaluate_energy_consumption`. Mix it into
    anything whose cost is worth reporting; it says nothing about what that
    thing does.

    Costs are keyed by the method that spent them, because a single running
    total cannot answer "how long does indexing take" once several stages are
    instrumented. Nested stages are inclusive: a method that calls another
    counts the callee's time inside its own. Call them separately when you want
    the two apart.
    """

    def __init__(self,
                 time_it: bool=True,
                 evaluate_energy_consumption: bool=True,
                 ):
        self._time_it = time_it
        self._evaluate_energy_consumption = evaluate_energy_consumption
        self.costs: dict[str, dict] = {}

    def _stage(self, name: str) -> dict:
        return self.costs.setdefault(name, {"calls": 0, "seconds": 0.0,
                                            "kwh": 0.0, "co2_g": 0.0})

    def record_time(self, stage: str, seconds: float) -> None:
        entry = self._stage(stage)
        entry["calls"] += 1
        entry["seconds"] += seconds

    def record_energy(self, stage: str, kwh: float, co2_g: float) -> None:
        entry = self._stage(stage)
        entry["kwh"] += kwh
        entry["co2_g"] += co2_g

    def cost_report(self) -> dict:
        """Per-stage totals plus a mean per call, ready to be written out."""
        return {stage: {**values,
                        "seconds_per_call": values["seconds"] / max(values["calls"], 1)}
                for stage, values in self.costs.items()}
