from abc import ABC, abstractmethod
from pathlib import Path
import random
from collections import defaultdict
from typing import Union
import torch
from torch.utils.data import Dataset, Sampler, DataLoader
from torchvision.transforms import v2
from transformers.image_utils import load_image
from typing import Tuple
import numpy as np

RANDOM_SEED = 42


def make_transform(resize_size: int = 224):
    to_tensor = v2.ToImage()
    resize = v2.Resize((resize_size, resize_size), antialias=True)
    to_float = v2.ToDtype(torch.float32, scale=True)
    normalize = v2.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    )
    return v2.Compose([to_tensor, resize, to_float, normalize])


def extract_paths_and_labels(directory: Path) -> Tuple[list, list]:
    paths = []
    labels = []
    for class_dir in directory.iterdir():
        if class_dir.is_dir():
            for img_path in class_dir.iterdir():
                paths.append(img_path.as_posix())
                labels.append(int(class_dir.name))

    return paths, labels


class DataPreparator():

    def __init__(self,
                original_data_path: Path,
                augmented_data_path: Path,
                random_seed: int=RANDOM_SEED,
                shuffle: bool=True
                ):
        self.original_data_path = original_data_path
        self.augmented_data_path = augmented_data_path
        self.random_seed = random_seed
        self.classes = self._get_classes()
        self.num_classes = len(self.classes)
        if shuffle:
            self._shuffle_classes()


    def _get_classes(self):
        if not self.original_data_path.exists():
            print(f"Error: Original data directory not found at {self.original_data_path}")
            return None
        
        all_classes = sorted([d.name for d in self.original_data_path.iterdir() if d.is_dir()])
        num_classes = len(all_classes)
        
        if num_classes == 0:
            print(f"Error: No class folders found in {self.original_data_path}")
            return None

        print(f"Found {num_classes} total classes.")
        return all_classes
        

    def _shuffle_classes(self) -> list:
        random.seed(self.random_seed)
        np.random.seed(self.random_seed)

        shuffled_classes = np.random.permutation(self.classes)
        self.classes = shuffled_classes
    

    def _get_classes_splits(self, classes: set, train_ratio: int, val_ratio: int) -> Tuple[set, set, set]:
        num_train = int(self.num_classes * train_ratio)
        num_val = int(self.num_classes * val_ratio)
        
        # Ensure at least 1 class in each split if ratios are small
        num_train = max(1, num_train)
        num_val = max(1, num_val)
        train_classes = set(classes[:num_train])
        val_classes = set(classes[num_train : num_train + num_val])
        test_classes = set(classes[num_train + num_val:])
        
        print(f"Splitting classes: {len(train_classes)} Train / {len(val_classes)} Val / {len(test_classes)} Test\n")

        return train_classes, val_classes, test_classes
    

    def _create_dataset_splits(
        self,
        train_classes: set,
        val_classes: set,
        test_classes: set,
        k_query: int,
    ):
        """
        Creates train, gallery, and query splits based on class-level separation.
        """

        random.seed(self.random_seed)
        np.random.seed(self.random_seed)

        
        train_paths, train_labels = [], []
        gallery_paths, gallery_labels = [], []
        val_query_paths, val_query_labels = [], []
        test_query_paths, test_query_labels = [], []


        print("Processing Train classes...")
        for class_name in train_classes:
            
            aug_class_dir = self.augmented_data_path / class_name
            aug_images = [str(p) for p in aug_class_dir.glob('*.*')]
            train_paths.extend(aug_images)
            train_labels.extend([int(class_name)] * len(aug_images))
            
            orig_class_dir = self.original_data_path / class_name
            orig_images = [str(p) for p in orig_class_dir.glob('*.*')]
            gallery_paths.extend(orig_images)
            gallery_labels.extend([int(class_name)] * len(orig_images))

        
        print("Processing Validation classes...")
        for class_name in val_classes:
            orig_class_dir = self.original_data_path / class_name
            all_class_images = [str(p) for p in orig_class_dir.glob('*.*')]
            random.shuffle(all_class_images)
            
            val_query_paths.extend(all_class_images[:k_query])
            val_query_labels.extend([int(class_name)] * k_query)
            
            gallery_paths.extend(all_class_images[k_query:])
            gallery_labels.extend([int(class_name)] * (len(all_class_images) - k_query))

        
        print("Processing Test classes...")
        for class_name in test_classes:
            orig_class_dir = self.original_data_path / class_name
            all_class_images = [str(p) for p in orig_class_dir.glob('*.*')]
            random.shuffle(all_class_images)
            
            test_query_paths.extend(all_class_images[:k_query])
            test_query_labels.extend([int(class_name)] * k_query)
            
            gallery_paths.extend(all_class_images[k_query:])
            gallery_labels.extend([int(class_name)] * (len(all_class_images) - k_query))

        print("\n--- Data Split Summary ---")
        print(f"Training Loader:   {len(train_paths):>5} samples from {len(train_classes)} classes (Augmented)")
        print(f"Gallery Loader:    {len(gallery_paths):>5} samples from {self.num_classes} classes (Original)")
        print(f"Val Query Loader:  {len(val_query_paths):>5} samples from {len(val_classes)} classes (Original)")
        print(f"Test Query Loader: {len(test_query_paths):>5} samples from {len(test_classes)} classes (Original)")
        

        data_splits = {
            "train": (train_paths, train_labels),
            "gallery": (gallery_paths, gallery_labels),
            "val_query": (val_query_paths, val_query_labels),
            "test_query": (test_query_paths, test_query_labels)
        }

        return data_splits


    def _get_tvt_splits(self, train_ratio, val_ratio):
        num_train = int(self.num_classes * train_ratio)
        num_val = int(self.num_classes * val_ratio)
        
        # Ensure at least 1 class in each split if ratios are small
        num_train = max(1, num_train)
        num_val = max(1, num_val)
        train_classes = set(self.classes[:num_train])
        val_classes = set(self.classes[num_train : num_train + num_val])
        test_classes = set(self.classes[num_train + num_val:])

        return train_classes, val_classes, test_classes


    def _get_k_fold_splits(self, split_num: int, val_ratio: float) -> Tuple[set, set, set]:
        val_start_idx = int(split_num * val_ratio * self.num_classes)
        val_stop_idx = int((split_num+1) * val_ratio * self.num_classes)
        val_classes = set(self.classes[val_start_idx:val_stop_idx])
        train_classes = set(self.classes).difference(val_classes)

        return train_classes, val_classes


    def get_k_folds(self,
                    k: int,
                    n_query: int=1):
        folds = []
        val_ratio = 1/k
        for split_num in range(k):
            print(f"================ PROCESSING SPLIT {split_num} ================")
            train_split, val_split = self._get_k_fold_splits(split_num, val_ratio)
            fold = self._create_dataset_splits(train_split, val_split, set(), n_query)
            fold.pop("test_query")
            folds.append(fold)
            print("\n")
        
        return folds


    def train_val_test_split(self, train_ratio: float, val_ratio: float, n_query: int):
        train_classes, val_classes, test_classes = self._get_tvt_splits(train_ratio, val_ratio)
        splits = self._create_dataset_splits(train_classes, val_classes, test_classes, n_query)

        return splits



class ImageCollectionDataset(ABC):
    def __init__(self, images_paths: list[Path], labels: list[int], transform: v2.Compose=None):
        self.images_paths = images_paths
        self.labels = labels
        self.transform = transform if transform else make_transform()
        self.class_to_indices = defaultdict(list)
        for idx, label in enumerate(labels):
            self.class_to_indices[label].append(idx)

    @abstractmethod
    def __len__(self):
        pass

    @abstractmethod
    def __getitem__(self, idx: Union[int, list[int]]):
        pass


# ==== Dataset simple ====
class CachedCollection(ImageCollectionDataset, Dataset):
    def __init__(self, images_paths: list[Path], labels: list[int], transform: v2.Compose=None):
        super().__init__(images_paths, labels, transform)
        self.images_instances = self.load_images()

    def load_images(self) -> list:
        images_instances = []
        for path in self.images_paths:
            images_instances.append(v2.Resize((224, 224))(load_image(path)))
        return images_instances

    def __len__(self):
        return len(self.images_instances)

    def __getitem__(self, idx):
        img = self.transform(self.images_instances[idx])
        label = self.labels[idx]
        return img, label
    

class LazyLoadCollection(ImageCollectionDataset, Dataset):
    def __init__(self, images_paths: list[Path], labels: list[int], transform: v2.Compose=None):
        super().__init__(images_paths, labels, transform)

    def __len__(self):
        return len(self.images_paths)

    def __getitem__(self, idx):
        img = load_image(self.images_paths[idx])
        if self.transform:
            img = self.transform(img)
        label = self.labels[idx]
        return img, label
    

# ==== Sampler pour P classes × K images ====
class PKSampler(Sampler):
    def __init__(self, dataset, P, K):
        self.P = P  # classes per batch
        self.K = K  # images per class
        self.dataset = dataset
        self.class_to_indices = self.dataset.class_to_indices
        self.classes = list(self.class_to_indices.keys())

    def __iter__(self):
        for _ in range(len(self)):
            # choose P classes at random
            selected_classes = random.sample(self.classes, self.P)
            batch_indices = []
            for c in selected_classes:
                indices = self.class_to_indices[c]
                # choose K samples from these indices
                chosen = []
                start_idx = random.randint(0, len(indices)-1)
                for i in range(self.K):
                    chosen.append(indices[(start_idx+i) % len(indices)])
                batch_indices.extend(chosen)
            yield batch_indices

    def __len__(self):
        return len(self.class_to_indices) // self.P
