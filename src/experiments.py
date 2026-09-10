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
from src.data import CachedCollection, DataPreparator, PKSampler
from src.distances.fusion import RRFBasedFusion
from src.distances.index import BinaryStrategy, DenseIndex, SparseIndex, TFIDFStrategy
from src.distances.kernels import (BhattacharyyaKernel, BinaryJaccardKernel,
                                   EuclidianDistanceKernel)
from src.eval import ConfusionArray, Recall
from src.config import load_config
from src.extractors import (BagOfVisualWords, DocTRTextExtractor, HSVExtractor,
                            MockRun, N_ROTATION_VIEWS, OrbFeatureExtractor,
                            SIFTFeatureExtractor, RotationAveraged, SiameseDino,
                            Whitened)
from src.feature_stores import InMemoryStore
from src.rerankers import HSVReranker, ORBReranker
from src.types import RetrievalChannel

EXTRACTORS = {"hsv": HSVExtractor, "orb": OrbFeatureExtractor,
              "sift": SIFTFeatureExtractor, "doctr": DocTRTextExtractor}
# "<name>-bovw" wraps a raw extractor in a visual vocabulary;
# "siamese" is built separately, needing a model config and optionally weights
KERNELS = {"bhattacharyya": BhattacharyyaKernel, "euclidean": EuclidianDistanceKernel,
           "jaccard": BinaryJaccardKernel}
WEIGHTINGS = {"binary": BinaryStrategy, "tfidf": TFIDFStrategy}
RERANKERS = {"hsv": HSVReranker, "orb": ORBReranker}
# where the whitening is applied: nowhere, on the descriptors an extractor hands
# out, or folded into the projection head's initialisation.
#
# head_init has yet to win anything. At full rank it *is* post -- identical
# fold for fold across both datasets, with and without rotation averaging -- and
# its only reason to exist is to ask whether training the head beats starting it
# at the whitening. Measured on cls at 224px over seg_gray, it does not: 0.912
# R@1 frozen against 0.872 trained, degrading further as the learning rate
# rises. Worse, the head start itself is absorbed. On 304 images it was worth
# 6.8 points over a random head; on 1216 with a P x K sampler, 0.853 either way.
# A linear preconditioner of a linear layer buys speed, not a better optimum,
# so it pays exactly where training is least affordable. Kept because it is the
# only way to ask the question again if the backbone, the loss or the scale
# change.
WHITEN_MODES = ("none", "post", "head_init")
# which images a descriptor-only fit may see; labels never leave the train split
FIT_CORPORA = ("train", "train+gallery", "all")


@dataclass
class DataSpec:
    path: str
    augmented_path: str = None  # source of the train split; defaults to `path`
    sampler_p: int = None       # classes per training batch; None iterates in file order
    sampler_k: int = None       # images per class in a training batch
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
    is_trainable: bool = False  # train the extractor on the fold's train split
    vocabulary_size: int = 256  # <name>-bovw -- number of visual words
    whiten: str = "none"        # none | post | head_init -- see WHITEN_MODES
    whiten_eps_rel: float = 0.05
    rotation_tta: bool = False  # average over the four 90-degree rotations
    pooling: str = None         # extractor: siamese -- cls | gem | avg, else the model config's
    projection_head_size: int = None  # extractor: siamese -- 0 drops the head
    config: str = None          # extractor: siamese -- path to the model config
    checkpoint: str = None      # extractor: siamese -- weights to load, else the bare backbone

    def __post_init__(self):
        """Refuse combinations that would be silently ignored rather than run.

        A dropped flag is worse here than a crash: the record would carry
        `whiten` in its config and so earn its own fingerprint, while
        describing a run identical to the one without it. Two entries claiming
        to compare something, measuring the same thing.
        """
        if self.whiten not in WHITEN_MODES:
            raise ValueError(f"whiten must be one of {WHITEN_MODES}, got {self.whiten!r}")

        if self.whiten == "post" and (self.index != "dense" or self.kernel != "euclidean"):
            # whitening emits signed dense vectors: bhattacharyya needs
            # non-negative ones and jaccard needs binary, so neither survives it
            raise ValueError(
                f"whiten: post needs index: dense and kernel: euclidean "
                f"(whitening then euclidean is the Mahalanobis distance), "
                f"got index: {self.index}, kernel: {self.kernel}")

        if self.whiten == "head_init" and self.extractor != "siamese":
            raise ValueError(
                f"whiten: head_init folds the whitening into a projection head, "
                f"which only extractor: siamese has; got {self.extractor!r}. "
                f"Use whiten: post instead.")

    @property
    def needs_fit(self) -> bool:
        """Whether the channel must see a corpus before it can describe anything.

        Derived rather than declared: a vocabulary and a whitening always need
        fitting, so making the config say so again only creates a way to forget.
        `is_trainable` stays a genuine choice -- the same backbone serves frozen
        or trained -- so it is the one thing left to declare.
        """
        return (self.is_trainable or self.whiten != "none"
                or self.extractor.endswith("-bovw"))


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
    fit_corpus: str = "train"   # train | train+gallery | all -- see FIT_CORPORA
    smoothing_param: int = 10
    recall_k: list[int] = field(default_factory=lambda: [1, 3, 5])

    def __post_init__(self):
        if self.fit_corpus not in FIT_CORPORA:
            raise ValueError(f"fit_corpus must be one of {FIT_CORPORA}, got {self.fit_corpus!r}")

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


def build_extractor(spec: ChannelSpec):
    """Wrap the base extractor in whatever the spec asks for, innermost first.

    Rotation averaging sits below the whitening: the whitening is then fitted on
    the descriptors it will actually transform, rather than on single views it
    never sees again.

    `whiten: head_init` adds no wrapper at all -- the whitening lives inside the
    model as its head's initialisation, and its rotation averaging goes with
    it. Wrapping would put the head inside the per-view loop, so each view
    would be whitened and the four averaged afterwards; the whitening belongs
    on the descriptor the channel settles on, not on the views it discards.
    """
    extractor = _base_extractor(spec)
    if spec.rotation_tta and spec.whiten != "head_init":
        extractor = RotationAveraged(extractor)
    if spec.whiten == "post":
        extractor = Whitened(extractor, eps_rel=spec.whiten_eps_rel)
    return extractor


def _base_extractor(spec: ChannelSpec):
    if spec.extractor.endswith("-bovw"):
        base = EXTRACTORS[spec.extractor.removesuffix("-bovw")]()
        return BagOfVisualWords(base, vocabulary_size=spec.vocabulary_size)
    if spec.extractor != "siamese":
        return EXTRACTORS[spec.extractor]()

    if not spec.config:
        raise ValueError("extractor: siamese needs a `config:` pointing at a model config")

    # MockRun rather than a real one: an experiment should not open a W&B run per
    # fold, and the config it would read from may well name a project
    model = SiameseDino(load_config(spec.config), run=MockRun(),
                        pooling=spec.pooling,
                        projection_head_size=spec.projection_head_size,
                        trainable=spec.is_trainable,
                        whiten_head=spec.whiten == "head_init",
                        whiten_eps_rel=spec.whiten_eps_rel,
                        # head_init averages inside the model, below the head;
                        # every other mode leaves it to the RotationAveraged wrapper
                        rotation_views=(N_ROTATION_VIEWS
                                        if spec.rotation_tta and spec.whiten == "head_init"
                                        else 1))
    if spec.checkpoint:
        model.load_state_dict(torch.load(spec.checkpoint, map_location=model.device))
    model.eval()
    return model


def build_channel(spec: ChannelSpec) -> RetrievalChannel:
    kernel = KERNELS[spec.kernel]()
    if spec.index == "sparse":
        index = SparseIndex(kernel, WEIGHTINGS[spec.weighting]())
    else:
        index = DenseIndex(kernel)
    return RetrievalChannel(build_extractor(spec), index, weight=spec.weight,
                            is_trainable=spec.needs_fit)


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


def _train_loader(config: ExperimentConfig, paths: list, labels: list,
                  preprocessor: v2.Compose, collate) -> DataLoader:
    """The train split, batched so a triplet loss has triplets to mine.

    A triplet needs an anchor, a positive of its class and a negative of
    another, all inside one batch. Iterating in file order gives whatever the
    file order gives: four images per class and a batch of sixteen happens to
    yield four classes of four, which works by luck, while sixteen images per
    class yields one class and no negatives at all. The loss then mines zero
    triplets and training silently does nothing while every epoch still prints
    its line -- measured, an augmented split trained for ten epochs at two
    different learning rates and returned the same score to three decimals.

    `sampler_p` and `sampler_k` make the composition explicit, and together
    they set the batch: `batch_size` then applies to the gallery and the
    queries only. Left unset, the order is the file order, which is only safe
    when the dataset happens to interleave classes.
    """
    dataset = CachedCollection(paths, labels, preprocessor=preprocessor)
    if config.data.sampler_p and config.data.sampler_k:
        return DataLoader(dataset, collate_fn=collate,
                          batch_sampler=PKSampler(dataset, config.data.sampler_p,
                                                  config.data.sampler_k))
    return DataLoader(dataset, batch_size=config.data.batch_size, collate_fn=collate)


def fit_corpus_split(mode: str, fold: dict) -> tuple[list, list]:
    """The images a descriptor-only fit may see, per `fit_corpus`.

    Whitening and vocabularies read descriptors and never labels, so they are
    not confined to the labelled train split the way a triplet loss is. How far
    past it they may go is a methodological choice, not a detail:

    - `train` is always honest, and the smallest corpus. A covariance in a few
      hundred dimensions estimated on a few hundred images is what the
      `eps_rel` shrinkage exists to prop up.
    - `train+gallery` is reproducible in deployment -- the gallery is enrolled
      before any query arrives, so its covariance is knowable then too -- and
      uses no labels, which is what separates it from fitting an LDA on the
      gallery.
    - `all` adds the queries, and is transductive: the estimate then depends on
      the very queries it will be scored against, which no deployed system
      gets. It inflates recall without the deployed system benefiting.

    Whichever is chosen lands in the config, and so in the fingerprint, because
    two records fitted under different regimes are not comparable.
    """
    paths, labels = list(fold["train"][0]), list(fold["train"][1])
    if mode == "train":
        return paths, labels

    paths += list(fold["gallery"][0])
    labels += list(fold["gallery"][1])
    if mode == "all":
        paths += list(fold["val_query"][0])
        labels += list(fold["val_query"][1])
    return paths, labels


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
            # gallery and queries always come from `path`; the train split may come
            # from an augmented copy, which is the only split augmentation belongs in
            folds = DataPreparator(config.data.path,
                                   config.data.augmented_path or config.data.path,
                                   random_seed=seed).get_k_folds(
                config.data.k_folds, config.data.gallery_instances, config.data.n_query)

        for fold_index, fold in enumerate(folds):
            gallery_paths, gallery_labels = fold["gallery"]
            query_paths, query_labels = fold["val_query"]
            train_paths, train_labels = fold["train"]

            loaders = [DataLoader(CachedCollection(paths, labels, preprocessor=preprocessor),
                                  batch_size=config.data.batch_size, collate_fn=collate)
                       for paths, labels in ((gallery_paths, gallery_labels),
                                             (query_paths, query_labels))]

            engine = build_engine(config, preprocessor, progress=not quiet)
            metrics = [Recall(recall_k=config.recall_k), ConfusionArray()]
            with contextlib.redirect_stdout(io.StringIO()) if quiet else contextlib.nullcontext():
                if any(spec.needs_fit for spec in config.channels):
                    # two corpora: anything reading labels gets the train split
                    # alone, anything reading descriptors gets what fit_corpus
                    # allows -- see `fit_corpus_split`
                    train_loader = _train_loader(config, train_paths, train_labels,
                                                 preprocessor, collate)
                    corpus_paths, corpus_labels = fit_corpus_split(config.fit_corpus, fold)
                    corpus_loader = None if config.fit_corpus == "train" else DataLoader(
                        CachedCollection(corpus_paths, corpus_labels, preprocessor=preprocessor),
                        batch_size=config.data.batch_size, collate_fn=collate)
                    engine.fit(train_loader, corpus_loader)

                # indexing runs on its own so its cost stays separate from querying
                engine.prepare_gallery(loaders[0])
                scores = engine.evaluate(loaders[0], loaders[1], metrics)

            records.append({
                "experiment": config.name,
                "fingerprint": fingerprint,
                "seed": seed,
                "fold": fold_index,
                "gallery_size": len(gallery_paths),
                "train_size": len(train_paths),
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
