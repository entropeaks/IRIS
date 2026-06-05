from abc import ABC, abstractmethod
from typing import List, Dict, TypeAlias
import torch
import numpy as np

Score: TypeAlias = float | List[float]


class Metric(ABC):
    def __init__(self):
        pass
    
    @abstractmethod
    def compute() -> Score:
        pass


class Recall(Metric):
    def __init__(self, recall_k: List=[1, 3, 10]):
        self.recall_k = recall_k

    def compute(self, dists: torch.Tensor,
                query_labels: torch.Tensor,
                gallery_labels: torch.Tensor
                ) -> Dict:
        
        topk_indices = dists.topk(max(self.recall_k), largest=False).indices  # (num_queries, k)

        recall_at_k = {}
        for k in self.recall_k:
            correct = 0
            for i, q_label in enumerate(query_labels):
                retrieved_labels = gallery_labels[topk_indices[i, :k]]
                if (retrieved_labels == q_label).any():
                    correct += 1
            recall_at_k[f"recall@{k}"] = correct / len(query_labels)

        return recall_at_k


class ConfusionArray:
    
    def compute(self,
                dists: torch.Tensor,
                query_labels: torch.Tensor,
                gallery_labels: torch.Tensor
                ) -> list[list]:
        
        topk_indices = dists.topk(1, largest=False).indices

        l = []
        for i, q_label in enumerate(query_labels):
            retrieved_label = gallery_labels[topk_indices[i]]
            l.append((q_label, retrieved_label))



""" class ModelReport:

    def __init__(self, models: dict[str, BaseModel], fit_model: bool=False):
        self.models = models
        self.fit_model = fit_model
        self.col = ["training time",
                    "training energy",
                    "inference time",
                    "inference energy",
                    "metrics",
                    "total test time",
                    "total test energy"
                    ]

    def generate_report(self, dataset_path: str, export_path: str, metric: Metric=Recall()) -> pd.DataFrame:
        data = []

        EXPERIMENT_DATA_PATH = Path("../data/augmented_data16")
        ORIGINAL_DATA_PATH = Path(dataset_path)

        data_preparator = DataPreparator(ORIGINAL_DATA_PATH, EXPERIMENT_DATA_PATH, 42)

        training_size = 0.7
        data_splits = data_preparator.train_val_test_split(train_ratio=0, val_ratio=1, n_query=1)
        train_paths, train_labels = data_splits["train"]
        gallery_paths, gallery_labels = data_splits["gallery"]
        val_query_paths, val_query_labels = data_splits["val_query"]
        test_query_paths, test_query_labels = data_splits["test_query"]
        print(gallery_paths)
        print(val_query_paths)

        browser = Browser(ORIGINAL_DATA_PATH)
        gallery_paths, gallery_labels, query_paths, query_labels = browser.sample_leave_k_out(1)

        RESIZE_SIZE = 224
        
        print("Loading dataloaders...")
        gallery_dataset = LazyLoadCollection(gallery_paths, gallery_labels, transform=make_transform(RESIZE_SIZE))
        gallery_dataloader = DataLoader(gallery_dataset, batch_size=32)

        val_dataset = LazyLoadCollection(query_paths, query_labels, transform=make_transform(RESIZE_SIZE))
        val_dataloader = DataLoader(val_dataset, batch_size=32)

        if self.fit_model:
            for model in self.models:
                if self.models[model].issubclass(DeepModel):

        for i, model in enumerate(self.models):
            print(f"------------------- MODEL {i} ---------------------")
            model_data = {}
            estimator = self.models[model]
            metrics = estimator.evaluate(gallery_dataloader, val_dataloader, metric)
            print(metrics)
            model_data["evaluation time"] = estimator.time
            model_data["evaluation_g.eq.co2"] = estimator.carbon
            model_data["evaluation_kWh"] = estimator.energy
            estimator.find_nearest_neighbors(val_query_paths[0], 3)
            model_data["inference time"] = estimator.time
            model_data["inference_g.eq.co2"] = estimator.carbon
            model_data["inference_kWh"] = estimator.energy
            model_data.update(metrics)
            model_data["store size (mb)"] = estimator.gallery_store.size()

            data.append(model_data)

            print("--------------------------------------------------")

        df = pd.DataFrame(data, index=list(self.models.keys()))

        return df """