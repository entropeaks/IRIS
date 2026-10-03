# IRIS

**IRIS** ***(Instance Retrieval & Identification System)*** is a modular backend framework designed to build and evaluate visual instance retrieval systems.

The project explores how different visual representations — from classical computer vision descriptors to deep metric learning embeddings — can be integrated into a unified architecture for similarity search and instance identification.

🏗️ *Note: this project is currently under active development. All recent updates and documentation are located in the `dev` branch until the first stable version is released.*

## 🏛️ Architecture

IRIS is built around a modular pipeline separating key components of a retrieval system:

* **Feature Extractors**: Pluggable modules producing visual descriptors (SIFT, ORB, deep embeddings).
* **Similarity Kernels**: Distance strategies adapted to different feature spaces.
* **Feature Fusion Models**: Hybrid approaches combining heterogeneous representations.
* **Evaluation Engine**: A unified benchmarking interface to compare multiple approaches on the same dataset.

This design allows rapid experimentation with different modeling strategies while keeping the system architecture clean and extensible.

## 🔬 Running an evaluation

An experiment is a YAML file. Running it appends one record per split; a separate
script reads those records afterwards.

```bash
pip install -e .
cp config/config.example.yaml config/config.yaml   # only for the deep channels

python scripts/evaluate.py configs/orb_bovw.yaml configs/orb_bovw_sift.yaml --out results/records.jsonl
python scripts/report.py results/records.jsonl --against orb-bovw
```

A config names the data, the retrieval channels and how their rankings combine:

```yaml
name: orb-bovw
data:
  path: /path/to/dataset          # one directory per class
  k_folds: 4
  seeds: [42, 43, 44]
channels:
  - extractor: orb-bovw           # hsv | orb | sift | doctr | siamese | <name>-bovw
    index: dense                  # dense | sparse
    kernel: euclidean             # bhattacharyya | euclidean | jaccard
    whiten: post                  # none | post | head_init
    weight: 1.0                   # its say during fusion
recall_k: [1, 3, 5]
```

The report gives recall, cost and a paired comparison. Adding SIFT geometric
verification on top of the top 10 candidates, over 40 draws:

```
experiment                    draws         R@1         R@3         R@5
orb-bovw+sift                    40  83.0+/-6.2  90.9+/-5.1  92.8+/-4.6
orb-bovw                         40  73.4+/-7.9  84.7+/-6.4  89.1+/-5.5

experiment                          evaluate (ms)   prepare_gallery (ms)
orb-bovw                                    165.6                  654.3
orb-bovw+sift                              2479.0                  662.2

Paired against orb-bovw
experiment                     shared   gap R@1   win/loss
orb-bovw+sift                      40     +9.53       37/2
```

Three things this layout buys:

* **Records are per split, not averaged.** Two configurations are compared only
  on the draws they share, because a single split here carries several points of
  recall noise. Averaging throws that pairing away: `+9.53` above is the mean of
  40 differences measured on the same data, and `37/2` says SIFT won 37 of them
  and lost 2 — far more convincing than two averages that happen to differ.
* **Cost sits beside accuracy.** Those 9.5 points cost 15x the query time, 166 ms
  to 2479 ms. Indexing barely moves, because the work is per comparison, not per
  gallery image. Whether that trade is worth taking depends on the application,
  and no recall column alone would surface it.
* **The report never runs anything.** A new question costs a read of the records
  rather than another evaluation.

Pass `--groups` a file of known near-duplicate families and the report also says
how many errors fell inside one, which separates *the model confused two classes*
from *these two classes are the same picture in different colours*.

## 🕸️ Technical Explorations

The framework enables experimentation on several axes:

* classical CV vs deep metric learning
* sparse vs dense representations
* feature-level fusion
* trade-offs between accuracy and computational cost

## 🎯 Case Study: Fine-Grained Visual Recognition

IRIS was initially developed to tackle a challenging fine-grained visual recognition problem: identifying bottle caps with nearly identical visual signatures.

These objects present several challenges:

* arbitrary 360° rotations
* specular reflections from metallic surfaces
*  subtle chromatic differences

The framework was used to benchmark different approaches ranging from classical Bag of Visual Words pipelines to deep metric learning models, uncovering their strenghts and weaknesses.