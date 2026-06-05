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