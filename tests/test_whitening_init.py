"""Whitening as a head initialisation, and the fit plumbing that feeds it.

Nothing here downloads a backbone. The claim that folding a whitening into a
projection head leaves retrieval unchanged is linear algebra, and testing it on
descriptors rather than on images is what makes it a test rather than a run.

Runnable as `python -m tests.test_whitening_init`, or under pytest.
"""

import numpy as np
import torch
from torch import nn

from src.experiments import ChannelSpec, build_extractor, fit_corpus_split
from src.extractors import RotationAveraged, Whitened
from src.postprocess import ZCAWhitening
from src.types import FeatureExtractor


def anisotropic_descriptors(n: int = 400, dim: int = 48, seed: int = 0) -> np.ndarray:
    """Descriptors with a wide eigenvalue spread, so whitening has work to do."""
    rng = np.random.default_rng(seed)
    mixing = rng.normal(size=(dim, dim)) @ np.diag(rng.uniform(0.1, 8.0, dim))
    return rng.normal(size=(n, dim)) @ mixing + rng.normal(size=dim) * 3


def pairwise(features: np.ndarray) -> np.ndarray:
    return np.linalg.norm(features[:, None, :] - features[None, :, :], axis=-1)


def l2(features: np.ndarray) -> np.ndarray:
    return features / (np.linalg.norm(features, axis=1, keepdims=True) + 1e-12)


def test_full_rank_projection_preserves_zca_distances():
    """ZCA and PCA-whitening differ by a rotation, which no distance can see.

    This is what lets a head hold the whitening: `whitener_` is square and
    cannot be cut to a narrower head, its PCA factor can, and at full rank the
    two are the same measurement.
    """
    features = anisotropic_descriptors()
    whitener = ZCAWhitening(eps_rel=0.05).fit(features)

    projection, mean = whitener.projection()
    folded = l2((features - mean) @ projection)

    assert projection.shape == (features.shape[1], features.shape[1])
    # float32, because ZCAWhitening.transform returns float32
    assert np.abs(pairwise(whitener.transform(features)) - pairwise(folded)).max() < 1e-5


def test_projection_keeps_the_strongest_directions():
    """`eigh` sorts ascending; truncating from the front would keep only noise."""
    features = anisotropic_descriptors()
    whitener = ZCAWhitening(eps_rel=0.0).fit(features)

    full, _ = whitener.projection()
    truncated, _ = whitener.projection(8)

    assert truncated.shape == (features.shape[1], 8)
    assert np.allclose(truncated, full[:, :8])

    # the kept directions must carry more variance than the dropped ones
    centered = features - features.mean(axis=0)
    variances = np.var(centered @ np.linalg.qr(full)[0], axis=0)
    assert variances[:8].min() > variances[8:].max()


def test_linear_head_reproduces_the_whitening():
    """`x @ W.T + b` with `W = P.T`, `b = -m @ P` is the whitening exactly."""
    features = anisotropic_descriptors()
    whitener = ZCAWhitening(eps_rel=0.05).fit(features)
    head_size = 16
    projection, mean = whitener.projection(head_size)

    head = nn.Linear(features.shape[1], head_size)
    with torch.no_grad():
        head.weight.copy_(torch.as_tensor(projection.T, dtype=head.weight.dtype))
        head.bias.copy_(torch.as_tensor(-mean @ projection, dtype=head.bias.dtype))

    produced = head(torch.as_tensor(features, dtype=torch.float32)).detach().numpy()
    expected = (features - mean) @ projection
    assert np.abs(produced - expected).max() < 1e-3


class RecordingExtractor(FeatureExtractor):
    """A trainable extractor whose descriptors change once it has been fitted.

    The shift is the point: it makes the order of a nested fit observable
    instead of a matter of trust.
    """

    trainable = True

    def __init__(self):
        self.fitted = False
        self.fit_calls = 0

    def get_features(self, imgs_arrays_rgb):
        offset = 100.0 if self.fitted else 0.0
        rng = np.random.default_rng(len(imgs_arrays_rgb))
        return list(rng.normal(size=(len(imgs_arrays_rgb), 6)) + offset)

    def fit(self, dataloader=None, corpus_dataloader=None):
        self.fit_calls += 1
        self.fitted = True


def fake_loader(n_batches: int = 4, batch_size: int = 8):
    return [([np.zeros((4, 4, 3), dtype=np.uint8)] * batch_size,
             list(range(batch_size))) for _ in range(n_batches)]


def test_wrapper_delegates_fit_to_what_it_wraps():
    """The bug this branch exists for: an outer wrapper swallowing the fit.

    Stacked as `build_extractor` stacks them, so the test fails the same way
    the harness did -- a fit that reaches the whitening and stops there.
    """
    inner = RecordingExtractor()
    stacked = Whitened(RotationAveraged(inner))

    stacked.fit(fake_loader())
    assert inner.fit_calls == 1, "the inner extractor was never fitted"


def test_whitening_is_fitted_after_the_extractor_it_wraps():
    """Order, not just delegation: a whitening fitted on pre-training
    descriptors would be calibrated on a distribution that no longer exists."""
    inner = RecordingExtractor()
    whitened = Whitened(inner)

    whitened.fit(fake_loader())

    assert inner.fitted
    # descriptors jump by 100 once fitted; the centre proves which it saw
    assert whitened.whitener.mean_.mean() > 50


def test_untrainable_inner_extractor_is_left_alone():
    class Frozen(RecordingExtractor):
        trainable = False

    inner = Frozen()
    Whitened(inner).fit(fake_loader())
    assert inner.fit_calls == 0


def test_needs_fit_is_derived_not_declared():
    """A whitening always needs fitting, so the config should not have to say so."""
    assert ChannelSpec(extractor="hsv", kernel="bhattacharyya").needs_fit is False
    assert ChannelSpec(extractor="siamese", config="c.yaml", whiten="post",
                       kernel="euclidean").needs_fit is True
    assert ChannelSpec(extractor="orb-bovw", index="sparse",
                       kernel="jaccard").needs_fit is True
    assert ChannelSpec(extractor="siamese", config="c.yaml",
                       is_trainable=True).needs_fit is True


def test_incompatible_whitening_is_refused_not_dropped():
    """Silently ignoring the flag would give the record its own fingerprint
    while describing a run identical to the one without it."""
    for kwargs in ({"kernel": "bhattacharyya"}, {"index": "sparse", "kernel": "jaccard"}):
        try:
            ChannelSpec(extractor="hsv", whiten="post", **kwargs)
        except ValueError:
            continue
        raise AssertionError(f"whiten: post accepted with {kwargs}")

    try:
        ChannelSpec(extractor="hsv", whiten="head_init")
    except ValueError:
        pass
    else:
        raise AssertionError("whiten: head_init accepted on an extractor with no head")

    try:
        ChannelSpec(extractor="hsv", whiten="yes")
    except ValueError:
        pass
    else:
        raise AssertionError("an unknown whiten mode was accepted")


def test_head_init_keeps_the_rotation_averaging_below_the_head():
    """Wrapping a whitened head puts it inside the per-view loop.

    Each view would then be whitened and the four averaged afterwards, which
    amplifies each view's own noise before averaging can suppress it -- and
    leaves the whitening transforming single views though it was fitted on
    averages. Measured on cls at 224px: 0.343 R@1 wrapped, 0.754 averaged
    first. So head_init must average inside the model, and must not be wrapped.
    """
    import src.experiments as harness

    base = RecordingExtractor()
    original = harness._base_extractor
    harness._base_extractor = lambda spec: base
    try:
        post = harness.build_extractor(
            ChannelSpec(extractor="hsv", whiten="post", kernel="euclidean",
                        rotation_tta=True))
        head_init = harness.build_extractor(
            ChannelSpec(extractor="siamese", config="c.yaml", whiten="head_init",
                        kernel="euclidean", rotation_tta=True))
    finally:
        harness._base_extractor = original

    assert isinstance(post, Whitened) and isinstance(post.extractor, RotationAveraged)
    # nothing wraps the model: it averages below its own head instead
    assert head_init is base


def test_fit_corpus_widens_without_ever_adding_labels_to_training():
    fold = {"train": (["t1", "t2"], [0, 1]),
            "gallery": (["g1"], [2]),
            "val_query": (["q1"], [2])}

    assert fit_corpus_split("train", fold) == (["t1", "t2"], [0, 1])
    assert fit_corpus_split("train+gallery", fold)[0] == ["t1", "t2", "g1"]
    assert fit_corpus_split("all", fold)[0] == ["t1", "t2", "g1", "q1"]
    # the fold's own lists must survive being read
    assert fold["train"][0] == ["t1", "t2"]


if __name__ == "__main__":
    # Apple's Accelerate BLAS raises spurious FP flags on well-scaled matmuls,
    # exactly as src.postprocess documents; they are noise in a test log
    np.seterr(over="ignore", invalid="ignore", divide="ignore")
    failures = 0
    for name, test in sorted(globals().items()):
        if not name.startswith("test_") or not callable(test):
            continue
        try:
            test()
            print(f"  ok   {name}")
        except Exception as err:
            failures += 1
            print(f"  FAIL {name}: {type(err).__name__}: {err}")
    print(f"\n{'all green' if not failures else f'{failures} failing'}")
    raise SystemExit(1 if failures else 0)
