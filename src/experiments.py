"""Describe a retrieval experiment in YAML, and run it into a record per draw.

One config, one run, many records. Each record carries the configuration that
produced it, the split it was measured on, its recall and its cost, so a report
can group and compare afterwards without anything being re-run.

Records are per (seed, fold) rather than averaged, because the useful comparison
between two configurations is paired -- same seeds, same folds -- and an average
throws away the pairing. On this dataset a single draw carries several points of
noise, so an unpaired difference of a few points means nothing.
"""

from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json

import dacite
import torch
import yaml
from torch.utils.data import DataLoader
from torchvision.transforms import v2

from src.core.engine import SearchEngine
from src.data import CachedCollection, DataPreparator
from src.distances.fusion import RRFBasedFusion
from src.distances.index import BinaryStrategy, DenseIndex, SparseIndex, TFIDFStrategy
from src.distances.kernels import (BhattacharyyaKernel, BinaryJaccardKernel,
                                   EuclidianDistanceKernel)
from src.eval import ConfusionArray, Recall
from src.extractors import DocTRTextExtractor, HSVExtractor, OrbFeatureExtractor
from src.feature_stores import InMemoryStore
from src.rerankers import HSVReranker, ORBReranker
from src.types import RetrievalChannel

EXTRACTORS = {"hsv": HSVExtractor, "orb": OrbFeatureExtractor, "doctr": DocTRTextExtractor}
KERNELS = {"bhattacharyya": BhattacharyyaKernel, "euclidean": EuclidianDistanceKernel,
           "jaccard": BinaryJaccardKernel}
WEIGHTINGS = {"binary": BinaryStrategy, "tfidf": TFIDFStrategy}
RERANKERS = {"hsv": HSVReranker, "orb": ORBReranker}


@dataclass
class DataSpec:
    path: str
    k_folds: int = 4
    seeds: list[int] = field(default_factory=lambda: [42])
    gallery_instances: int = 1
    n_query: int = 1
    resize: int = 224
    batch_size: int = 32


@dataclass
class ChannelSpec:
    extractor: str
    index: str = "dense"
    kernel: str = "bhattacharyya"
    weighting: str = "binary"
    weight: float = 1.0


@dataclass
class RerankerSpec:
    type: str
    top_k_candidates: int = 10


@dataclass
class ExperimentConfig:
    name: str
    data: DataSpec
    channels: list[ChannelSpec]
    reranker: RerankerSpec = None
    smoothing_param: int = 10
    recall_k: list[int] = field(default_factory=lambda: [1, 3, 5])

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ExperimentConfig":
        raw = yaml.safe_load(Path(path).read_text())
        return dacite.from_dict(cls, raw, config=dacite.Config(strict=True))

    def fingerprint(self) -> str:
        """Stable id for this configuration, ignoring which seeds it ran on.

        Lets a report group records that describe the same setup, and tell two
        setups apart even when someone reused a name.
        """
        payload = asdict(self)
        payload["data"].pop("seeds")
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:12]


def build_channel(spec: ChannelSpec) -> RetrievalChannel:
    kernel = KERNELS[spec.kernel]()
    if spec.index == "sparse":
        index = SparseIndex(kernel, WEIGHTINGS[spec.weighting]())
    else:
        index = DenseIndex(kernel)
    return RetrievalChannel(EXTRACTORS[spec.extractor](), index, weight=spec.weight)


def build_engine(config: ExperimentConfig, preprocessor: v2.Compose,
                 progress: bool=True) -> SearchEngine:
    reranker = RERANKERS[config.reranker.type]() if config.reranker else None
    return SearchEngine(
        preprocessor,
        [build_channel(spec) for spec in config.channels],
        InMemoryStore(),
        RRFBasedFusion(smoothing_param=config.smoothing_param),
        reranker=reranker,
        top_k_candidates=config.reranker.top_k_candidates if config.reranker else 50,
        time_it=True,
        evaluate_energy_consumption=False,
        progress=progress,
    )


def run(config: ExperimentConfig, quiet: bool = True) -> list[dict]:
    """Evaluate one configuration on every (seed, fold) draw, one record each."""
    import contextlib, io

    preprocessor = v2.Resize((config.data.resize, config.data.resize))
    fingerprint = config.fingerprint()
    records = []

    def collate(batch):
        return [item[0] for item in batch], [item[1] for item in batch]

    for seed in config.data.seeds:
        sink = io.StringIO() if quiet else None
        with contextlib.redirect_stdout(sink) if quiet else contextlib.nullcontext():
            folds = DataPreparator(config.data.path, config.data.path, random_seed=seed).get_k_folds(
                config.data.k_folds, config.data.gallery_instances, config.data.n_query)

        for fold_index, fold in enumerate(folds):
            gallery_paths, gallery_labels = fold["gallery"]
            query_paths, query_labels = fold["val_query"]

            loaders = [DataLoader(CachedCollection(paths, labels, preprocessor=preprocessor),
                                  batch_size=config.data.batch_size, collate_fn=collate)
                       for paths, labels in ((gallery_paths, gallery_labels),
                                             (query_paths, query_labels))]

            engine = build_engine(config, preprocessor, progress=not quiet)
            metrics = [Recall(recall_k=config.recall_k), ConfusionArray()]
            with contextlib.redirect_stdout(io.StringIO()) if quiet else contextlib.nullcontext():
                # indexing runs on its own so its cost stays separate from querying
                engine.prepare_gallery(loaders[0])
                scores = engine.evaluate(loaders[0], loaders[1], metrics)

            records.append({
                "experiment": config.name,
                "fingerprint": fingerprint,
                "seed": seed,
                "fold": fold_index,
                "gallery_size": len(gallery_paths),
                "n_queries": len(query_paths),
                "recall": {str(k): scores[f"recall@{k}"] for k in config.recall_k},
                "confusion": scores["confusion"],
                "costs": engine.cost_report(),
                "config": asdict(config),
                "recorded": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            })
            if not quiet:
                print(f"  seed {seed} fold {fold_index}: "
                      f"{ {k: v for k, v in scores.items() if k != 'confusion'} }")

    return records


def append_records(records: list[dict], path: str | Path) -> None:
    """Append one JSON object per line, so runs accumulate instead of replacing."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")


def load_records(path: str | Path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
