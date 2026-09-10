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

    def __init__(self, eps_rel: float = 0.05):
        self.eps_rel = eps_rel
        self.mean_ = None
        self.whitener_ = None
        self.eigenvalues_ = None
        self.eigenvectors_ = None

    def fit(self, features: np.ndarray) -> "ZCAWhitening":
        features = np.asarray(features, dtype=np.float64)
        self.mean_ = features.mean(axis=0)
        centered = features - self.mean_
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
        if self.whitener_ is None:
            raise RuntimeError("ZCAWhitening.transform called before fit")
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            projected = (np.asarray(features, dtype=np.float64) - self.mean_) @ self.whitener_
        projected /= np.linalg.norm(projected, axis=-1, keepdims=True) + 1e-12
        return projected.astype(np.float32)

    def fit_transform(self, features: np.ndarray) -> np.ndarray:
        return self.fit(features).transform(features)
