import numpy as np
from scipy.stats import rankdata


class RRFBasedFusion():
    """Reciprocal rank fusion of several distance matrices.

    Each channel ranks the gallery independently and votes with 1/(k + rank), so
    a channel's opinion counts most where it is confident -- near the top of its
    own ranking -- and a candidate it places 40th rather than 30th barely moves
    the result.

    `smoothing_param` sets how fast that vote decays with depth. The value of 60
    usually quoted comes from web-scale runs; on a gallery of a hundred entries
    1/(60 + rank) is nearly flat, so every rank votes almost equally and the
    damping is lost. Keep it small when the gallery is small.

    RRF weights every channel equally. It suits channels of comparable strength;
    one much weaker than the others drags the result down whatever the smoothing.
    """

    def __init__(self, smoothing_param: int=10):
        self._smoothing_param = smoothing_param


    def normalize_distances(self, dists: np.ndarray) -> np.ndarray:
        """Rank each row independently; rank 1 is the closest gallery entry."""
        return np.apply_along_axis(rankdata, 1, np.asarray(dists, dtype=float))


    def fuse(self, feature_ranks: list) -> np.ndarray:
        """Combine per-channel ranks into a distance matrix (lower is better).

        Expects the output of `normalize_distances`, not raw distances.
        """
        if not feature_ranks:
            raise ValueError("fuse() needs at least one channel")

        score = np.zeros_like(np.asarray(feature_ranks[0], dtype=float))
        for ranks in feature_ranks:
            score += 1.0 / (self._smoothing_param + np.asarray(ranks, dtype=float))

        # RRF scores are similarities; negate so callers keep treating low as close
        return -score
