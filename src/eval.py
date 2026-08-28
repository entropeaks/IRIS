from abc import ABC, abstractmethod
from typing import List, Dict, TypeAlias
import torch

Score: TypeAlias = float | List[float]


class Metric(ABC):
    def __init__(self):
        pass
    
    @abstractmethod
    def compute(self, dists, query_labels, gallery_labels) -> Score:
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


class ConfusionArray(Metric):
    """Which class each query was actually matched to, for error analysis.

    Answers what recall cannot: not how often retrieval failed but what it
    reached for instead. Pairs, not a matrix, since most of a square over a
    hundred classes would be zeros.
    """

    def compute(self,
                dists: torch.Tensor,
                query_labels: torch.Tensor,
                gallery_labels: torch.Tensor
                ) -> dict:

        top_indices = dists.topk(1, largest=False).indices.squeeze(1)
        return {"confusion": [[int(true), int(gallery_labels[retrieved])]
                              for true, retrieved in zip(query_labels, top_indices)]}
