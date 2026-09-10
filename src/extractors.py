
from pathlib import Path
import warnings
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, List, Tuple
import uuid
import numpy as np
import cv2
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.config import Config
from src.types import FeatureExtractor
from src.utils import set_device
from src.eval import Metric

# doctr, transformers and wandb cost roughly 2.7s, 1.0s and 0.5s to import and
# are each needed by a single class here. Importing them where they are used
# keeps `import src.rerankers` -- which only needs OpenCV -- from paying
# for all three.
if TYPE_CHECKING:
    from wandb import Run

    
class DocTRTextExtractor(FeatureExtractor):

    def __init__(self):
        from doctr.models import ocr_predictor

        super().__init__()
        # half precision is a CUDA-only win here; several ops fall back or fail
        # on cpu and mps, so keep full precision off CUDA
        device = set_device("auto")
        self.ocr = ocr_predictor(
            det_arch="db_mobilenet_v3_large",
            reco_arch="crnn_mobilenet_v3_small",
            pretrained=True
        ).to(device)
        if device.type == "cuda":
            self.ocr = self.ocr.half()
    
    def get_features(self, imgs_arrays_rgb: list[np.ndarray]) -> list[list[str]]:
        result = self.ocr(imgs_arrays_rgb)
        
        return [
            [
                word.value.lower()
                for block in page.blocks
                for line in block.lines
                for word in line.words
            ]
            for page in result.pages
        ]
    

class OrbFeatureExtractor(FeatureExtractor):
    """Raw ORB descriptors: a variable-length (n_keypoints, 32) array per image.

    Left raw on purpose. `ORBReranker` matches these pairwise with a Hamming
    matcher and needs the descriptors themselves; wrap this in
    `BagOfVisualWords` to get something an index can hold instead.
    """

    def __init__(self, n_features: int=500):
        self.trainable = False
        self.orb = cv2.ORB_create(nfeatures=n_features)

    def get_features(self, imgs_arrays_rgb: list[np.ndarray]) -> list[np.ndarray]:
        descriptors = []
        for img in imgs_arrays_rgb:
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            _, described = self.orb.detectAndCompute(gray, None)
            descriptors.append(described)
        return descriptors


class ExtractorWrapper(FeatureExtractor):
    """Base for extractors that decorate another one.

    It owns the `fit` delegation. A wrapper that fits something of its own --
    a vocabulary, a whitening -- still sits on top of an extractor that may
    have its own training to do, and forgetting to pass `fit` down silently
    leaves the inner model untrained while everything downstream still runs.
    Subclasses override `_fit_self`, never `fit`, so the delegation cannot be
    dropped by omission.

    The inner extractor is fitted first. Each stage is then estimated on the
    descriptors it will actually transform, rather than on ones the stage below
    it will stop producing the moment it is trained.
    """

    def __init__(self, extractor: FeatureExtractor):
        self.extractor = extractor
        self.trainable = True

    def fit(self, dataloader: DataLoader, corpus_dataloader: DataLoader=None) -> None:
        if self.extractor.trainable:
            self.extractor.fit(dataloader, corpus_dataloader)
        self._fit_self(corpus_dataloader if corpus_dataloader is not None else dataloader)

    def _fit_self(self, dataloader: DataLoader) -> None:
        """Fit whatever this wrapper adds, on descriptors from the fitted inner
        extractor. Override; the default has nothing of its own to estimate."""


class BagOfVisualWords(ExtractorWrapper):
    """Quantises another extractor's local descriptors into a fixed vocabulary.

    ORB and SIFT give a variable number of descriptors per image, which no index
    here can hold: `DenseIndex` cannot stack rows of different lengths and
    `SparseIndex` needs discrete terms. Clustering a corpus's descriptors turns
    each into the id of its nearest centroid, so an image becomes a bag of visual
    words -- the shape `SparseIndex` was written for, TF-IDF included.

    A wrapper rather than a base class, because the raw descriptors are still
    wanted elsewhere: `ORBReranker` matches them pairwise and would have nothing
    to match if quantising replaced describing.

    Note k-means minimises squared euclidean distance, which suits SIFT and only
    approximates ORB, whose binary descriptors live in Hamming space. That is the
    usual trade in these pipelines: what matters downstream is that similar
    patches land in the same cluster, not where the centroid sits.
    """

    def __init__(self, extractor: FeatureExtractor, vocabulary_size: int=256,
                 seed: int=42, max_fit_descriptors: int=200_000):
        super().__init__(extractor)
        self.vocabulary_size = vocabulary_size
        self.seed = seed
        self.max_fit_descriptors = max_fit_descriptors
        self.kmeans = None

    def get_features(self, imgs_arrays_rgb: list[np.ndarray]) -> list[list[int]]:
        if self.kmeans is None:
            raise RuntimeError(
                "BagOfVisualWords has no vocabulary; call fit() on the corpus first, "
                "or mark its channel is_trainable so the engine does")

        # centroids are fitted in float32; sklearn would promote uint8 descriptors
        # to float64 and then refuse the mismatch
        return [[] if described is None
                else self.kmeans.predict(described.astype(np.float32)).tolist()
                for described in self.extractor.get_features(imgs_arrays_rgb)]

    def _fit_self(self, dataloader: DataLoader) -> None:
        """Cluster the corpus's descriptors into the visual vocabulary."""
        from sklearn.cluster import MiniBatchKMeans

        pool, collected = [], 0
        for images, _ in dataloader:
            for described in self.extractor.get_features(images):
                if described is None:
                    continue
                pool.append(described)
                collected += len(described)
            if collected >= self.max_fit_descriptors:
                break

        if not pool:
            raise RuntimeError(f"{type(self.extractor).__name__} found no keypoint "
                               f"anywhere in the fitting set")

        descriptors = np.vstack(pool).astype(np.float32)
        if len(descriptors) > self.max_fit_descriptors:
            rng = np.random.default_rng(self.seed)
            descriptors = descriptors[rng.choice(len(descriptors),
                                                 self.max_fit_descriptors, replace=False)]

        # more centroids than descriptors would leave empty words in the vocabulary
        k = min(self.vocabulary_size, len(descriptors))
        self.kmeans = MiniBatchKMeans(n_clusters=k, random_state=self.seed,
                                      n_init="auto", batch_size=4096).fit(descriptors)


class Whitened(ExtractorWrapper):
    """Wraps an extractor, whitening the descriptors it produces.

    Equalising the covariance of a descriptor set suppresses the few
    high-variance directions -- illumination, glare, pose -- that otherwise
    dominate euclidean distance. On frozen backbones it is worth more than any
    other single choice: measured over 40 paired draws, 28.7 -> 85.9 R@1 on
    MobileNetV3-Small and 71.1 -> 94.0 on DINOv3 ViT-S.

    A wrapper for the same reason `BagOfVisualWords` is one: whitening is a
    separate concern from describing, and the raw descriptors are still wanted
    by rerankers that match them pairwise.

    Whitening a descriptor that a trained head also produces is the same
    correction applied twice; `SiameseDino.init_head_from_whitening` folds it
    into the head instead. This wrapper is the form for extractors with no head
    to fold it into.

    The transform is fitted, so a channel using it must be marked trainable. It
    consumes descriptors only, never labels, so it may be fitted on a corpus
    wider than the labelled train split -- see `whiten_fit_on` in the harness.
    """

    def __init__(self, extractor: FeatureExtractor, eps_rel: float=0.05):
        super().__init__(extractor)
        self.eps_rel = eps_rel
        self.whitener = None

    def get_features(self, imgs_arrays_rgb: list[np.ndarray]) -> list[np.ndarray]:
        described = np.asarray(self.extractor.get_features(imgs_arrays_rgb), dtype=np.float64)
        described = described.reshape(len(described), -1)
        if self.whitener is None:
            raise RuntimeError(
                "Whitened has no transform; call fit() on the corpus first, "
                "or mark its channel is_trainable so the engine does")
        return list(self.whitener.transform(described))

    def _fit_self(self, dataloader: DataLoader) -> None:
        from src.postprocess import ZCAWhitening

        pool = []
        for images, _ in dataloader:
            described = np.asarray(self.extractor.get_features(images), dtype=np.float64)
            pool.append(described.reshape(len(described), -1))

        if not pool:
            raise RuntimeError("nothing to fit the whitening on")
        self.whitener = ZCAWhitening(eps_rel=self.eps_rel).fit(np.vstack(pool))


N_ROTATION_VIEWS = 4


def average_over_rotations(describe, imgs_arrays_rgb: list[np.ndarray],
                           n_views: int=N_ROTATION_VIEWS) -> np.ndarray:
    """Average `describe`'s descriptors over the first `n_views` 90-degree rotations.

    Free-standing rather than a method so `SiameseDino.pool_batch` can mirror
    it on pooled tokens: the two must agree, since a channel may average above
    the head or below it and the whitening's placement depends on which.

    Each view is L2-normalised before averaging so one view cannot dominate by
    magnitude, and the mean is normalised again.
    """
    total = None
    for turns in range(n_views):
        views = (imgs_arrays_rgb if turns == 0
                 else [np.rot90(img, turns).copy() for img in imgs_arrays_rgb])
        described = np.asarray(describe(views), dtype=np.float64)
        described = described.reshape(len(described), -1)
        described /= np.linalg.norm(described, axis=1, keepdims=True) + 1e-12
        total = described if total is None else total + described

    total /= np.linalg.norm(total, axis=1, keepdims=True) + 1e-12
    return total


class RotationAveraged(ExtractorWrapper):
    """Averages an extractor's descriptors over the four 90-degree rotations.

    The caps sit at arbitrary angles, and a descriptor that is not rotation
    invariant sees a different image each time one is turned. Averaging the four
    lossless rotations -- np.rot90 moves pixels without resampling, so nothing is
    interpolated away -- gives a descriptor that no longer depends on how the cap
    happened to land.

    Stateless of itself, so `_fit_self` stays empty: unlike the vocabulary or the
    whitening, there is nothing here to estimate from a corpus. It costs four
    backbone passes per image, which is the whole of its price.

    Whitening after this rather than before makes no measurable difference --
    whitening is affine, so it commutes with the mean up to the per-view
    renormalisation.
    """

    def __init__(self, extractor: FeatureExtractor, n_views: int=N_ROTATION_VIEWS):
        super().__init__(extractor)
        self.n_views = n_views

    def get_features(self, imgs_arrays_rgb: list[np.ndarray]) -> list[np.ndarray]:
        return list(average_over_rotations(self.extractor.get_features,
                                           imgs_arrays_rgb, self.n_views))


class SIFTFeatureExtractor(FeatureExtractor):

    def __init__(self, min_match_count: int=10):
        self.trainable = False
        self.sift = cv2.SIFT_create()
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        self.flann = cv2.FlannBasedMatcher(index_params, search_params)
        self.min_match_count = min_match_count
    
    def get_features(self, imgs_arrays_rgb: list[np.ndarray]) -> list[np.ndarray]:
        """Raw SIFT descriptors, one (n_keypoints, 128) array per image.

        Takes images like every other extractor. `compute_distances` below needs
        the keypoints too, so it detects them itself rather than widening what
        this returns.
        """
        descriptors = []
        for img in imgs_arrays_rgb:
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            _, described = self.sift.detectAndCompute(gray, None)
            descriptors.append(described)
        return descriptors

    def detect(self, img_array_rgb: np.ndarray):
        """Keypoints and descriptors for one image, for pairwise matching."""
        gray = cv2.cvtColor(img_array_rgb, cv2.COLOR_RGB2GRAY)
        return self.sift.detectAndCompute(gray, None)
    
    def compute_distances(self, feat1: Tuple, feat2: Tuple) -> int: 
        kp1, des1 = feat1
        kp2, des2 = feat2

        if des1 is None or des2 is None or len(des1) < 2 or len(des2) < 2:
            return 1.0

        matches = self.flann.knnMatch(des1, des2, k=2)

        good_matches = []
        for m, n in matches:
            if m.distance < 0.7*n.distance:
                good_matches.append(m)

        if len(good_matches) > self.min_match_count:
            src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1,1,2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1,1,2)

            mask: np.ndarray
            _, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            
            if mask is None:
                return 1.0
            
            inliers_count = np.sum(mask)
            
            return 1.0 / (inliers_count + 1)

        return 1.0
    

class HSVExtractor(FeatureExtractor):

    def __init__(self, hist_size: list=[50, 60]):
        self.trainable = False
        self._hist_size = hist_size


    def get_features(self, imgs_arrays_rgb: list[np.ndarray]):
        features = []
        for img_rgb in imgs_arrays_rgb:
            img_gw = self._apply_gray_world(img_rgb)
            img_hsv = cv2.cvtColor(img_gw, cv2.COLOR_RGB2HSV)
            mask = self._create_dynamic_mask(img_hsv)

            hist_size = self._hist_size
            ranges = [0, 180, 0, 256]
            
            hist = cv2.calcHist([img_hsv], [0, 1], mask, hist_size, ranges)
            cv2.normalize(hist, hist, alpha=1, beta=0, norm_type=cv2.NORM_L1)
            features.append(hist)

        return features


    def _apply_gray_world(self, rgb_img): 
        """
        Applique l'algorithme Gray World pour annuler la dominante de couleur de l'éclairage.
        Utilise des opérations matricielles pour l'optimisation des performances.
        """
        # Conversion en float32 pour éviter les dépassements (overflow) lors des calculs
        r, g, b = cv2.split(rgb_img.astype(np.float32))
        
        avg_r = np.mean(r)
        avg_g = np.mean(g)
        avg_b = np.mean(b)
        
        # Sécurité pour éviter la division par zéro sur une image totalement noire
        if avg_r == 0 or avg_g == 0 or avg_b == 0:
            return rgb_img
            
        avg_gray = (avg_r + avg_g + avg_b) / 3.0
        
        # Application des facteurs d'échelle d'illumination et bornage [0, 255]
        r = np.clip(r * (avg_gray / avg_r), 0, 255)
        g = np.clip(g * (avg_gray / avg_g), 0, 255)
        b = np.clip(b * (avg_gray / avg_b), 0, 255)
        
        result = cv2.merge([r, g, b])

        return result.astype(np.uint8)
    

    def _create_dynamic_mask(self, hsv_img):
        """
        Génère un masque binaire dynamique excluant les reflets spéculaires et les ombres.
        """
        h, s, v = cv2.split(hsv_img)
        
        # Identification dynamique de la luminance maximale de l'image courante
        v_max = np.max(v)
        
        # 1. Masque des reflets : Très lumineux (> 90% du max local) ET peu saturé (< 30)
        # Les opérations bitwise d'OpenCV sont écrites en C, idéales pour la scalabilité
        highlight_mask = cv2.bitwise_and(
            (v > 0.9 * v_max).astype(np.uint8),
            (s < 30).astype(np.uint8)
        ) * 255
        
        # 2. Masque des ombres : Valeur de luminosité extrêmement faible (bruit capteur)
        shadow_mask = (v < 20).astype(np.uint8) * 255
        
        # 3. Masque final : On garde les pixels qui ne sont NI des reflets, NI des ombres
        bad_pixels = cv2.bitwise_or(highlight_mask, shadow_mask)
        valid_mask = cv2.bitwise_not(bad_pixels)
        
        return valid_mask
    



class Pooling(ABC):
    """Reduces a transformer's token sequence to one vector per image.

    `n_prefix` counts the tokens before the patches -- CLS plus any register
    tokens -- so a pooling that wants patches only knows where they start.
    """

    @abstractmethod
    def __call__(self, hidden_states: torch.Tensor, n_prefix: int) -> torch.Tensor: ...


class ClsPool(Pooling):
    """The CLS token, which the backbone trained to summarise the image."""

    def __call__(self, hidden_states: torch.Tensor, n_prefix: int) -> torch.Tensor:
        return hidden_states[:, 0, :]


class AvgPool(Pooling):
    """Mean of the patch tokens."""

    def __call__(self, hidden_states: torch.Tensor, n_prefix: int) -> torch.Tensor:
        return hidden_states[:, n_prefix:, :].mean(dim=1)


class GemPool(Pooling):
    """Generalised mean of the patch tokens, emphasising the strongest responses.

    The clamp keeps the power well defined for p not an integer, and it is the
    reason this suits convolutional feature maps far better than transformer
    patch tokens: post-activation conv outputs are almost all non-negative, while
    patch tokens are freely signed, so the clamp flattens roughly half of every
    token to the floor. Measured on a frozen ViT it costs about 30 points of R@1
    against taking the CLS token.
    """

    def __init__(self, p: float=3.0, eps: float=1e-6):
        self.p = p
        self.eps = eps

    def __call__(self, hidden_states: torch.Tensor, n_prefix: int) -> torch.Tensor:
        patches = hidden_states[:, n_prefix:, :]
        return patches.clamp(min=self.eps).pow(self.p).mean(dim=1).pow(1.0 / self.p)


POOLINGS = {"cls": ClsPool, "gem": GemPool, "avg": AvgPool}


class MockRun:
    def __getattr__(self, name):
        # Retourne une fonction qui ne fait rien pour n'importe quel nom de méthode
        return lambda *args, **kwargs: None


class SiameseDino(FeatureExtractor, nn.Module):
    def __init__(self, config: Config, run: "Run"=None,
                 pooling: str=None, projection_head_size: int=None,
                 trainable: bool=False, whiten_head: bool=False,
                 whiten_eps_rel: float=0.05, rotation_views: int=1):
        """Backbone embedder with a projection head, the backbone frozen by default.

        `run` receives an existing W&B run. Without one, a run is opened only if
        `base.wandb_project_name` is configured; otherwise metrics go to a no-op
        sink, so the model can be built offline and in tests.

        `pooling` and `projection_head_size` override the model config, so an
        experiment can sweep them without editing it. A size of 0 leaves the
        pooled tokens alone, which is what a frozen backbone wants: an untrained
        head is a random projection, and training one is a separate decision from
        choosing a backbone.

        `trainable` says whether `fit` should train this model or leave it as a
        frozen descriptor. It is the extractor-level answer to the harness's
        `is_trainable`.

        `whiten_head` makes `fit` initialise the head from the corpus's
        whitening first -- see `init_head_from_whitening`. It is independent of
        `trainable`: initialising and never training gives the whitening on its
        own, which is the frozen-backbone baseline, and initialising then
        training asks whether training buys anything beyond it. Either implies
        a pass over a corpus, so `trainable` (what the engine consults) is the
        disjunction of the two.

        `rotation_views` averages the pooled tokens over that many 90-degree
        rotations before the head, instead of letting `RotationAveraged` wrap
        the whole model. The distinction matters only when the head carries a
        whitening: wrapped, the head sits inside the per-view loop, so each
        view is whitened and the four are averaged afterwards -- which
        amplifies each view's own noise before averaging can suppress it, and
        leaves the whitening transforming single views though it was fitted on
        averages. Averaging first puts the whitening back on the descriptor the
        channel actually settles on. Measured on cls at 224px, wrapped scores
        0.343 R@1 against 0.754 for the same whitening applied after averaging.
        """

        nn.Module.__init__(self)

        self._config = config
        from transformers import AutoModel, AutoImageProcessor

        self._backbone = AutoModel.from_pretrained(self._config.model.backbone_name)
        self.optimizer = None
        resize = {"height": self._config.train.resize.height, "width": self._config.train.resize.width}
        self._processor = AutoImageProcessor.from_pretrained(self._config.model.backbone_name, size=resize)
        
        #n_prefix is num_registers + 1 to take all patch tokens without CLS and register tokens
        self._n_prefix = self._backbone.config.num_register_tokens + 1
        
        self.pooling = POOLINGS[pooling or self._config.model.pooling]()
        head_size = (self._config.model.projection_head_size
                     if projection_head_size is None else projection_head_size)

        embedding_dim = self._backbone.config.hidden_size
        hidden_dim = self._config.model.hidden_dim
        dropout = self._config.model.dropout
        if head_size == 0:
            self.projection_head = nn.Identity()
        elif hidden_dim > 0:
            self.projection_head = nn.Sequential(
                nn.Linear(embedding_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, head_size))
        else:
            self.projection_head = nn.Sequential(
                nn.Linear(embedding_dim, head_size),
                nn.Dropout(dropout))
        self.output_dim = embedding_dim if head_size == 0 else head_size
        self.loss = nn.TripletMarginLoss(margin=self._config.train.margin, p=2)
        self.device = set_device(config.base.device)
        self.to(self.device)

        self._trains = trainable
        self._whiten_head = whiten_head
        self._whiten_eps_rel = whiten_eps_rel
        self.rotation_views = rotation_views
        self.trainable = trainable or whiten_head
        self.gallery_labels = None
        self._name = f"siamese-{uuid.uuid4().hex[:6]}"
        self.run = run or self._open_run()


    def set_optimizer(self, optimizer: torch.optim.Optimizer):
        """Override the optimizer `fit` would otherwise build for itself."""
        self.optimizer = optimizer


    def init_head_from_whitening(self, dataloader: DataLoader,
                                 eps_rel: float=0.05) -> "ZCAWhitening":
        """Set the projection head to the whitening of the corpus's descriptors.

        Whitening after a trained head is the same correction twice: the head is
        already a learned linear map, so a second linear map behind it has
        little left to add and cannot be fitted honestly anyway -- its corpus
        moves the moment the head trains. Folding the whitening into the head's
        *initialisation* removes both problems. There is one linear layer, and
        it starts where the whitening would have put it instead of at random.

        `nn.Linear` computes `x @ W.T + b`, and whitening computes
        `(x - m) @ P`, so `W = P.T` and `b = -m @ P`. The head is then exactly
        the whitening at step zero, and training moves away from it rather than
        from noise. Note `P` is the PCA-whitening factor, not the ZCA matrix:
        the two differ by a rotation, which no euclidean distance can see, and
        only the PCA form can be cut down to a head narrower than the backbone.

        The backbone being frozen, the descriptors this is fitted on are the
        ones the head keeps receiving for the whole of training -- which is what
        makes fitting it first correct here, and what makes it wrong for a
        whitening sitting after the head.

        Fitted on `pooled_features`, which already averages over rotations when
        the channel asked for them, so the whitening sees the descriptors the
        head will actually be given rather than single views it never keeps.
        """
        from src.postprocess import ZCAWhitening

        linear = self._single_linear_head()
        pool = [np.asarray(self.pooled_features(images), dtype=np.float64)
                for images, _ in dataloader]
        if not pool:
            raise RuntimeError("nothing to fit the head initialisation on")

        whitener = ZCAWhitening(eps_rel=eps_rel).fit(np.vstack(pool))
        projection, mean = whitener.projection(self.output_dim)
        with torch.no_grad():
            linear.weight.copy_(torch.as_tensor(projection.T, dtype=linear.weight.dtype))
            linear.bias.copy_(torch.as_tensor(-mean @ projection, dtype=linear.bias.dtype))
        return whitener


    def _single_linear_head(self) -> nn.Linear:
        """The head's one linear layer, or a refusal explaining why there isn't one."""
        layers = [module for module in self.projection_head.modules()
                  if isinstance(module, nn.Linear)]
        if len(layers) != 1:
            raise ValueError(
                f"a whitening initialisation needs a single-linear projection head, "
                f"this one has {len(layers)}. Set hidden_dim: 0 and "
                f"projection_head_size > 0, or use whiten: post to whiten the "
                f"descriptors after the extractor instead.")
        return layers[0]


    def _backbone_blocks(self) -> nn.ModuleList:
        """The backbone's transformer blocks, whatever this architecture calls them."""
        for path in ("layer", "encoder.layer", "encoder.layers", "blocks", "encoder.block"):
            module = self._backbone
            for name in path.split("."):
                module = getattr(module, name, None)
                if module is None:
                    break
            if isinstance(module, nn.ModuleList):
                return module
        raise ValueError(
            f"cannot find the transformer blocks of {self._config.model.backbone_name} "
            f"to unfreeze; set train.unfrozen_backbone_blocks: 0 to keep it frozen")


    def _freeze_backbone(self) -> list:
        """Freeze the backbone but for its last `unfrozen_backbone_blocks` blocks.

        `requires_grad` rather than leaving parameters out of the optimizer:
        omitting them stops the update but still computes and stores a gradient
        for the whole backbone on every batch, which is the bulk of the memory
        and a good part of the time.

        Returns the parameters that stay trainable, for the optimizer to adopt.
        """
        self._backbone.requires_grad_(False)
        n_blocks = self._config.train.unfrozen_backbone_blocks
        if n_blocks <= 0:
            return []

        blocks = self._backbone_blocks()[-n_blocks:]
        for block in blocks:
            block.requires_grad_(True)
        return [p for block in blocks for p in block.parameters()]


    def backbone_is_frozen(self) -> bool:
        return self._config.train.unfrozen_backbone_blocks <= 0


    def _build_optimizer(self) -> torch.optim.Optimizer:
        """Adam over the head, plus any backbone block left unfrozen.

        Built here rather than injected: which parameters exist and which are
        worth unfreezing is a property of the architecture, and an outside
        caller could only answer it by reaching into `_backbone`. `fit` builds
        one lazily, so a caller that wants another still sets it beforehand.
        """
        backbone_params = self._freeze_backbone()
        groups = [{"params": list(self.projection_head.parameters()),
                   "lr": self._config.train.head_lr}]
        if backbone_params:
            groups.append({"params": backbone_params,
                           "lr": self._config.train.backbone_lr})
        return torch.optim.Adam(groups, weight_decay=self._config.train.weight_decay)

    def _open_run(self):
        """Start a W&B run when one is configured, else return a no-op sink.

        wandb.init raises on bad credentials, an unreachable server or invalid
        arguments; training should not die because the logging sideline did. A
        failure is reported and metrics go to the sink instead.

        It can also hang rather than raise -- login has no timeout by default --
        which no except clause can catch. Set WANDB_MODE=offline or disabled to
        rule that out on a machine without network access.
        """
        if not self._config.base.wandb_project_name:
            return MockRun()

        import wandb

        try:
            return wandb.init(project=self._config.base.wandb_project_name,
                              entity=self._config.base.wandb_entity,
                              config=self._config)
        except Exception as err:
            warnings.warn(f"W&B run could not be started, metrics will not be logged: {err}",
                          RuntimeWarning, stacklevel=2)
            return MockRun()

    def set_run(self, run: "Run"):
        self.run = run

    def forward(self, **inputs):
        return self.head_forward(self.pool(**inputs))

    def pool(self, **inputs) -> torch.Tensor:
        """The pooled backbone tokens, before the projection head."""
        return self.pooling(self._backbone(**inputs).last_hidden_state, self._n_prefix)

    def pool_batch(self, imgs_arrays_rgb: list[np.ndarray]) -> torch.Tensor:
        """Pooled tokens for a batch of images, averaged over rotations if asked.

        The single place the backbone is run, so the whitening fit, the
        training epochs and the queries all describe an image the same way.
        Mirrors `average_over_rotations` -- L2 per view, mean, L2 again -- on
        pooled tokens rather than on finished descriptors, which is the whole
        point: it puts the averaging below the head instead of above it.
        """
        if self.rotation_views <= 1:
            inputs = self._processor(images=imgs_arrays_rgb, return_tensors="pt").to(self.device)
            return self.pool(**inputs)

        total = None
        for turns in range(self.rotation_views):
            views = (imgs_arrays_rgb if turns == 0
                     else [np.rot90(np.asarray(img), turns).copy() for img in imgs_arrays_rgb])
            inputs = self._processor(images=views, return_tensors="pt").to(self.device)
            pooled = F.normalize(self.pool(**inputs), p=2, dim=1)
            total = pooled if total is None else total + pooled
        return F.normalize(total, p=2, dim=1)

    def head_forward(self, pooled: torch.Tensor) -> torch.Tensor:
        """The head applied to already-pooled tokens.

        Split out of `forward` so a frozen backbone can be run once and its
        output reused: every training epoch then costs a matrix product per
        batch instead of a full backbone pass over images that cannot change.
        """
        x = self.projection_head(pooled)
        if self._config.model.normalize:
            x = F.normalize(x, p=2, dim=1)
        return x
    

    def to(self, device):
        self._backbone.to(device)
        self.projection_head.to(device)
        self.device = device
        return self
    
    
    @torch.no_grad
    def get_features(self, imgs_arrays_rgb: list[np.ndarray]):
        return self.head_forward(self.pool_batch(imgs_arrays_rgb)).cpu().numpy()


    @torch.no_grad
    def pooled_features(self, imgs_arrays_rgb: list[np.ndarray]) -> np.ndarray:
        """Descriptors as the head receives them: pooled, unprojected."""
        return self.pool_batch(imgs_arrays_rgb).cpu().numpy()


    def fit(self, dataloader: DataLoader, corpus_dataloader: DataLoader=None) -> None:
        """Train on the labelled split.

        `corpus_dataloader` feeds the head's whitening initialisation, which
        reads descriptors only. Training itself never sees it: triplet mining
        reads labels, so it may only ever have the train split.
        """
        if self._whiten_head:
            self.init_head_from_whitening(corpus_dataloader or dataloader,
                                          eps_rel=self._whiten_eps_rel)
        if not self._trains:
            self.eval()
            return

        self.train()
        if self.optimizer is None:
            self.optimizer = self._build_optimizer()

        print("--------------- Training Siamese model ---------------")
        cache = self._cache_pooled(dataloader) if self.backbone_is_frozen() else None
        for epoch in tqdm(range(self._config.train.epochs)):
            train_metrics = self._fit_one_epoch(dataloader, cache)
            print(f"Epoch {epoch+1}: {train_metrics}")
        # back to eval, or the gallery would be indexed with dropout live
        self.eval()


    @torch.no_grad
    def _cache_pooled(self, dataloader: DataLoader) -> list[tuple]:
        """Run the frozen backbone once and keep what it pooled.

        A frozen backbone returns the same descriptor for the same image at
        every epoch, so running it once per epoch spends the overwhelming
        majority of training on recomputing constants. Cached, an epoch costs
        one matrix product per batch.

        Cached in eval mode on purpose: a backbone in train mode would bake its
        dropout into the cache, and the same noise frozen across every epoch is
        worse than no noise at all.
        """
        self._backbone.eval()
        return [(self.pool_batch(images), labels) for images, labels in dataloader]


    def _epoch_batches(self, dataloader: DataLoader, cache: list[tuple]=None):
        """Yield `(embeddings, labels)` per batch, from cached descriptors if there are any."""
        if cache is not None:
            for pooled, labels in cache:
                yield self.head_forward(pooled), labels
            return

        for images, labels in dataloader:
            yield self.head_forward(self.pool_batch(images)), labels


    def fit_and_evaluate(self,
                         train_dataloader: DataLoader,
                         gallery_dataloader: DataLoader,
                         query_dataloader: DataLoader,
                         metric: Metric):
        if self.optimizer is None:
            self.optimizer = self._build_optimizer()
        best_score = 0.0
        best_metrics = {}
        for epoch in tqdm(range(self._config.train.epochs)):
            train_metrics = self._fit_one_epoch(train_dataloader)
            metrics = self._evaluate_new_iteration(gallery_dataloader, query_dataloader, metric)
            
            train_metrics.update(metrics)
            self.run.log(train_metrics)

            if self._model_improvement(metrics, best_score):
                best_metrics = metrics
                best_score = np.mean([score for score in best_metrics.values()])
                self.save()

        return best_metrics


    def _fit_one_epoch(self, train_dataloader: DataLoader, cache: list[tuple]=None) -> dict:
        self.train()
        if self.backbone_is_frozen():
            # a frozen backbone has no dropout or norm statistics to update
            self._backbone.eval()
        cumulative_loss = 0.0
        cumulative_pos_dist = 0.0
        cumulative_neg_dist = 0.0
        cumulative_triplets_count = 0
        for embeddings, labels in self._epoch_batches(train_dataloader, cache):
            triplets = self._mine_semi_hard_triplets_cdist(embeddings, labels)
            if not triplets:
                continue
            cumulative_triplets_count += len(triplets)
            anchor_indices, positive_indices, negative_indices = zip(*triplets)
            anchor_embeddings = embeddings[list(anchor_indices)]
            positive_embeddings = embeddings[list(positive_indices)]
            negative_embeddings = embeddings[list(negative_indices)]
            anchor_embeddings = anchor_embeddings.to(self.device)
            positive_embeddings = positive_embeddings.to(self.device)
            negative_embeddings = negative_embeddings.to(self.device)
            self.optimizer.zero_grad()
            triplet_loss = self.loss(anchor_embeddings, positive_embeddings, negative_embeddings)
            triplet_loss.backward()
            self.optimizer.step()
            cumulative_pos_dist += F.pairwise_distance(anchor_embeddings, positive_embeddings, p=2).mean().item()
            cumulative_neg_dist += F.pairwise_distance(anchor_embeddings, negative_embeddings, p=2).mean().item()
            cumulative_loss += triplet_loss.detach().cpu().item()

        cumulative_loss /= len(train_dataloader)
        cumulative_pos_dist /= len(train_dataloader)
        cumulative_neg_dist /= len(train_dataloader)

        return {"loss": cumulative_loss,
                "positive_dist": cumulative_pos_dist,
                "negative_dist": cumulative_neg_dist,
                "triplets_mined": cumulative_triplets_count}

    @torch.no_grad
    def _mine_semi_hard_triplets_cdist(self, embeddings: torch.Tensor, labels: np.ndarray) -> List[tuple]:
        """
        Vectorized semi-hard negative mining using torch.cdist.
        For each anchor, finds a random positive and all semi-hard negatives.
        A semi-hard negative `n` satisfies: d(a, p) < d(a, n) < d(a, p) + margin
        
        embeddings: torch.Tensor of shape (N, D)
        labels: list or np.array of length N
        margin: float, the margin used in the TripletLoss
        
        Returns: list of (anchor_idx, positive_idx, semi_hard_negative_idx)
        """
        if isinstance(embeddings, list):
            embeddings = torch.stack(embeddings)
        
        if isinstance(labels, torch.Tensor):
            labels = labels.cpu().numpy()
        elif isinstance(labels, Tuple) or isinstance(labels, list):
            labels = np.array(labels)

        n = embeddings.shape[0]
        # Calcule la matrice des distances au carré pour la stabilité, ou p=2 pour euclidienne
        dists = torch.cdist(embeddings, embeddings, p=2)
        
        triplets = []
        for anchor_idx in range(n):
            anchor_label = labels[anchor_idx]
            
            # Masques pour positifs et négatifs
            pos_mask = (labels == anchor_label) & (np.arange(n) != anchor_idx)
            pos_indices = np.where(pos_mask)[0]
            
            neg_mask = (labels != anchor_label)
            neg_indices = np.where(neg_mask)[0]
            
            if len(pos_indices) == 0 or len(neg_indices) == 0:
                continue
                
            # Itérer sur tous les positifs possibles pour cet ancre
            for positive_idx in pos_indices:
                pos_dist = dists[anchor_idx, positive_idx]

                # Condition 1: d(a, n) > d(a, p)
                cond1 = dists[anchor_idx, neg_indices] > pos_dist
                # Condition 2: d(a, n) < d(a, p) + margin
                cond2 = dists[anchor_idx, neg_indices] < (pos_dist + self._config.train.margin)
                
                semi_hard_neg_mask = cond1 & cond2
                
                semi_hard_indices = neg_indices[semi_hard_neg_mask.cpu().numpy()]
                
                for semi_hard_neg_idx in semi_hard_indices:
                    triplets.append((anchor_idx, positive_idx, semi_hard_neg_idx))
                    
        return triplets
    

    @torch.no_grad()
    def _evaluate_new_iteration(self, gallery_dataloader: DataLoader, query_dataloader, metric: Metric) -> dict:
        self.eval()
        gallery_embeddings, gallery_labels = self._compute_embeddings(gallery_dataloader)
        query_embeddings, query_labels = self._compute_embeddings(query_dataloader)

        dists = self.compute_distances(
            query_embeddings,
            gallery_embeddings
        )

        scores = metric.compute(dists, query_labels, gallery_labels)

        return scores
    
    
    def compute_distances(
        self,
        query_features: torch.Tensor,
        stored_features_batch: torch.Tensor
    ) -> torch.Tensor:
        """
        Calcule les distances entre query et un batch.
        Optimisé pour la vectorisation.
        """
        dists = torch.cdist(query_features, stored_features_batch, p=2)
        return dists
    

    def _compute_embeddings(self, dataloader: DataLoader):
        self.eval()

        embeddings, all_labels = [], []

        with torch.no_grad():
            for images, labels in dataloader:
                inputs = self._processor(images=images, return_tensors="pt").to(self.device)
                emb = self(**inputs)
                if not self._config.model.normalize:
                    emb = F.normalize(emb, p=2, dim=1)
                embeddings.append(emb.cpu())
                all_labels.append(labels)
        
        embeddings = torch.cat(embeddings, dim=0)
        all_labels = torch.cat(all_labels, dim=0)

        return embeddings, all_labels
    

    def _model_improvement(self, metrics: dict, best_score: float) -> bool:
        return np.mean([score for score in metrics.values()]) > best_score


    def save(self, name: str=None):
        """Write the state dict under the configured checkpoint directory."""
        directory = Path(self._config.base.model_checkpoints_path)
        directory.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), directory / f"{name or self._name}.pth")
            
