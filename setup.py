from setuptools import setup, find_packages

setup(
    name="diht",
    version="0.1.0",
    author="Timothée Coste",
    description="Deep Image Hashing Transformer (DIHT)",
    packages=find_packages(),
    install_requires=[
        "torch",
        "torchvision",
        "transformers",
        "tqdm",
        "numpy",
        "scipy",
        "pandas",
        "scikit-learn",
        "pyyaml",
        "dacite",
        "matplotlib",
        "pillow",
        "opencv-python",
        "ultralytics",
        "wandb",
        "codecarbon",
        "python-doctr",
    ],
    python_requires=">=3.9",
)
