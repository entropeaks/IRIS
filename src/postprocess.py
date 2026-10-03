"""Post-processing of descriptors, applied after a feature extractor has run.

Where `src.preprocess` transforms images before they reach a backbone, this
module transforms the embeddings that come out of one.
"""

import numpy as np


class ZCAWhitening:
    """Full-rank ZCA whitening of descriptors, with an eigenvalue floor.

    Whitening equalises the covariance of the descriptor set, which suppresses
    the few high-variance directions (illumination, glare, pose) that otherwise
    dominate euclidean distance. Fitting consumes descriptors only -- no class
    labels -- so it stays usable when no labelled corpus is available.

    Rather than truncating to the top components, every direction is kept and
    the gain applied to the low-variance tail is capped: eigenvalues below
    `eps_rel` times the mean eigenvalue are clipped up to that floor. Deleting
    the tail loses signal, whereas capping its gain keeps it without letting
    near-null directions blow up. The floor is expressed relative to the mean
    eigenvalue so that it transfers across backbones and feature scales; values
    between 0.02 and 0.1 behave equivalently on this data.
    """

    def __init__(self, eps_rel: float = 0.05, n_components: int = None):
        """`n_components` keeps only that many directions, strongest first.

        It also changes how the fit is computed, because the two cannot be
        separated cheaply. Full rank needs the `(d, d)` covariance and its
        eigendecomposition; truncated takes the SVD of the centred data
        instead, which costs `O(n^2 d)` rather than `O(d^3)` and never forms
        the covariance at all. At d=8192 that is the difference between a
        537 MB matrix and a few hundred rows.

        The two agree on every direction the data can measure. They differ on
        the ones it cannot: a `(n, d)` corpus pins down at most `n - 1`
        directions, and full rank keeps the remaining `d - n + 1` -- flooring
        their eigenvalues, so they are amplified by `floor**-0.5` although
        nothing was measured along them. That is survivable when `n` is close
        to `d` and ruinous when it is not, which is why anything wider than a
        backbone descriptor should pass this.
        """
        self.eps_rel = eps_rel
        self.n_components = n_components
        self.projection_ = None
        self.mean_ = None
        self.whitener_ = None
        self.eigenvalues_ = None
        self.eigenvectors_ = None

    def fit(self, features: np.ndarray) -> "ZCAWhitening":
        features = np.asarray(features, dtype=np.float64)
        self.mean_ = features.mean(axis=0)
        centered = features - self.mean_
        if self.n_components is not None:
            return self._fit_truncated(centered)
        # Apple's Accelerate BLAS raises spurious FP flags on well-scaled matmuls,
        # so silence them locally rather than touching global numpy state
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            covariance = centered.T @ centered / max(len(centered) - 1, 1)

            eigenvalues, eigenvectors = np.linalg.eigh(covariance)
            floor = self.eps_rel * np.trace(covariance) / covariance.shape[0]
            self.n_floored_ = int((eigenvalues < floor).sum())
            eigenvalues = np.clip(eigenvalues, floor, None)
            self.eigenvalues_, self.eigenvectors_ = eigenvalues, eigenvectors
            self.whitener_ = eigenvectors @ np.diag(eigenvalues ** -0.5) @ eigenvectors.T
        return self

    def _fit_truncated(self, centered: np.ndarray) -> "ZCAWhitening":
        """Fit from the SVD of the centred data, keeping `n_components`.

        The right singular vectors are the covariance's eigenvectors and the
        squared singular values its eigenvalues, so this is the same
        decomposition read off a matrix that is `n` rows tall instead of `d`
        wide. The floor is the same one, and the trace it is relative to is the
        sum of every eigenvalue, which the singular values give without the
        covariance ever being built.
        """
        n_samples, n_features = centered.shape
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            _, singular, components = np.linalg.svd(centered, full_matrices=False)

        eigenvalues = singular ** 2 / max(n_samples - 1, 1)
        floor = self.eps_rel * eigenvalues.sum() / n_features
        self.n_floored_ = int((eigenvalues[:self.n_components] < floor).sum())
        eigenvalues = np.clip(eigenvalues, floor, None)

        keep = min(self.n_components, len(eigenvalues))
        self.eigenvalues_ = eigenvalues[:keep][::-1]        # projection() reverses back
        self.eigenvectors_ = components[:keep].T[:, ::-1]
        self.projection_ = components[:keep].T * eigenvalues[:keep] ** -0.5
        return self

    def projection(self, n_components: int = None) -> tuple[np.ndarray, np.ndarray]:
        """The whitening as a `(descriptor_dim, n_components)` factor, and its centre.

        `whitener_` is the ZCA form `U L^-1/2 U^T`, square by construction: its
        trailing `U^T` rotates back into the original axes. That rotation is
        what makes it untruncatable -- and, for the same reason, irrelevant to
        euclidean distance. Dropping it leaves the PCA-whitening factor
        `U L^-1/2`, which whitens identically and can be cut to `n_components`
        columns, so it fits a projection head narrower than the descriptor.

        Returned strongest-direction-first. Truncating does discard the
        low-variance tail this class otherwise keeps -- see the note above on
        flooring rather than truncating -- so a full-rank request is the
        faithful one, and a narrower head trades that fidelity for its width.
        """
        if self.eigenvalues_ is None:
            raise RuntimeError("ZCAWhitening.projection called before fit")

        # eigh sorts eigenvalues ascending, so the strongest directions come last
        eigenvalues = self.eigenvalues_[::-1]
        eigenvectors = self.eigenvectors_[:, ::-1]

        k = len(eigenvalues) if n_components is None else n_components
        if not 0 < k <= len(eigenvalues):
            raise ValueError(f"n_components must be in 1..{len(eigenvalues)}, got {n_components}")

        return eigenvectors[:, :k] * eigenvalues[:k] ** -0.5, self.mean_

    def transform(self, features: np.ndarray) -> np.ndarray:
        matrix = self.whitener_ if self.projection_ is None else self.projection_
        if matrix is None:
            raise RuntimeError("ZCAWhitening.transform called before fit")
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            projected = (np.asarray(features, dtype=np.float64) - self.mean_) @ matrix
        projected /= np.linalg.norm(projected, axis=-1, keepdims=True) + 1e-12
        return projected.astype(np.float32)

    def fit_transform(self, features: np.ndarray) -> np.ndarray:
        return self.fit(features).transform(features)
