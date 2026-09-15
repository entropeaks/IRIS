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
from src.extractors import (BagOfVisualWords, OrbFeatureExtractor,
                            RotationAveraged, VisualWordHistogram, Whitened)
from src.postprocess import ZCAWhitening
from src.types import FeatureExtractor


def cornered_image(seed: int, size: int = 128) -> np.ndarray:
    """Random rectangles on a flat ground, so ORB has corners to detect.

    Uniform noise gives it nothing: the detector wants intensity corners, and
    smoothing turns pixel noise into flat grey.
    """
    rng = np.random.default_rng(seed)
    img = np.full((size, size, 3), 40, dtype=np.uint8)
    for _ in range(6):
        x, y = rng.integers(0, size - 30, 2)
        w, h = rng.integers(12, 30, 2)
        img[y:y + h, x:x + w] = rng.integers(120, 255, 3, dtype=np.uint8)
    return img


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


def test_truncated_fit_agrees_with_the_full_one():
    """Same decomposition, read off an (n, d) matrix instead of a (d, d) one.

    Where the corpus can measure every direction the two must agree exactly;
    the truncated path exists for width, not for a different answer. At d=8192
    with 300 samples it fits in 0.1s against 60s, and never builds the 537 MB
    covariance.
    """
    features = anisotropic_descriptors(n=400, dim=40)
    full = ZCAWhitening(eps_rel=0.05).fit(features)
    truncated = ZCAWhitening(eps_rel=0.05, n_components=40).fit(features)

    assert np.abs(pairwise(full.transform(features))
                  - pairwise(truncated.transform(features))).max() < 1e-5


def test_truncation_drops_directions_the_corpus_cannot_measure():
    """With n < d the covariance has rank n-1, so the rest is unmeasured.

    Full rank keeps them and amplifies them by the eigenvalue floor, which is
    how a whitening turns into a noise generator on a VLAD-width descriptor.
    """
    features = anisotropic_descriptors(n=30, dim=120)
    truncated = ZCAWhitening(eps_rel=0.05, n_components=16).fit(features)

    assert truncated.transform(features).shape == (30, 16)
    # the full-rank path keeps all 120, of which at most 29 were measured
    assert ZCAWhitening(eps_rel=0.05).fit(features).transform(features).shape == (30, 120)


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


def test_visual_words_count_into_a_fixed_length_vector():
    """The term list is ragged by construction; the histogram cannot be.

    A whitening has no covariance to estimate from rows of different lengths,
    and DenseIndex cannot stack them either, so this is what makes a dense
    bag-of-words channel possible at all.
    """
    images = [cornered_image(seed) for seed in range(12)]
    loader = [(images, list(range(len(images))))]

    histogram = VisualWordHistogram(OrbFeatureExtractor(), vocabulary_size=8)
    histogram.fit(loader)
    described = np.asarray(histogram.get_features(images))

    assert described.shape == (len(images), histogram.kmeans.n_clusters)
    assert np.allclose(np.linalg.norm(described, axis=1), 1.0)

    # the sparse form keeps its one-word-per-keypoint contract
    terms = BagOfVisualWords(OrbFeatureExtractor(), vocabulary_size=8)
    terms.fit(loader)
    assert all(isinstance(t, list) for t in terms.get_features(images))


def test_an_image_without_keypoints_does_not_divide_by_zero():
    histogram = VisualWordHistogram(OrbFeatureExtractor(), vocabulary_size=4)
    histogram.fit([([cornered_image(s) for s in range(8)], list(range(8)))])

    blank = np.zeros((128, 128, 3), dtype=np.uint8)   # ORB finds no corner here
    described = np.asarray(histogram.get_features([blank]))
    assert described.shape == (1, histogram.kmeans.n_clusters)
    assert np.all(np.isfinite(described)) and described.sum() == 0


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
