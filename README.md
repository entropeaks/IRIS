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

python scripts/evaluate.py configs/hsv.yaml configs/siamese.yaml --out results/records.jsonl
python scripts/report.py results/records.jsonl --against hsv --groups miscellaneous/groups.json
```

A config names the data, the retrieval channels and how their rankings combine:

```yaml
name: hsv
data:
  path: /path/to/dataset          # one directory per class
  k_folds: 4
  seeds: [42, 43, 44]
channels:
  - extractor: hsv                # hsv | orb | doctr | siamese
    index: dense                  # dense | sparse
    kernel: bhattacharyya         # bhattacharyya | euclidean | jaccard
    weight: 1.0                   # its say during fusion
recall_k: [1, 3, 5]
```

The report gives recall, cost and a paired comparison:

```
experiment                draws           R@1           R@3           R@5
siamese                      12    19.3+/-8.8      26.8+/-9.8      31.7+/-9.9
hsv                          12     8.5+/-4.8      20.6+/-6.0      27.8+/-6.2
hsv-reranked                 12     8.5+/-4.8      20.6+/-6.0      27.8+/-6.2

experiment                       evaluate (ms)  prepare_gallery (ms)
hsv                                       11.3                  40.5
hsv-reranked                             129.7                  41.5
siamese                                  253.5                1012.9

Paired against hsv, Wilcoxon signed-rank, * = p<0.05
experiment                shared   gap R@1   win/loss         p
hsv-reranked                  12     +0.00        0/0         1
siamese                       12    +10.81       10/1     0.002 *
```

Three things this layout buys:

* **Records are per split, not averaged.** Two configurations are compared on the
  draws they share, because a single split here carries several points of recall
  noise and an unpaired difference of a few points means nothing.
* **Cost sits beside accuracy.** Reranking HSV with HSV above costs 11× more per
  query for exactly nothing — a fact no recall column alone would surface.
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