from abc import ABC, abstractmethod
from typing import List, Dict, TypeAlias
import torch

Score: TypeAlias = float | List[float]


class Metric(ABC):
    def __init__(self):
        pass
    
    @abstractmethod
    def compute() -> Score:
        pass

#TODO define a type ConfusionMatrix to comply with for metrics

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
