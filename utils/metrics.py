from typing import Tuple

import torch
import numpy as np
from utils.eval import cluster_acc
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score
from sklearn.metrics import adjusted_rand_score as ari_score


def compute_metrics(preds: np.ndarray, labels: np.ndarray) -> Tuple[float, float, float]:
    acc = cluster_acc(labels, preds)
    nmi = nmi_score(labels, preds)
    ari = ari_score(labels, preds)

    return acc, nmi, ari


def calculate_prototypes(features, labels, num_classes):
    class_assignments = labels.view(labels.shape[0], 1).expand(-1, features.shape[1])

    one_hot = torch.nn.functional.one_hot(labels, num_classes)
    labels_count = one_hot.sum(dim=0)

    prototypes = torch.zeros((num_classes, features.shape[1]), dtype=features.dtype, device=features.device)
    prototypes.scatter_add_(0, class_assignments, features)
    prototypes = prototypes / labels_count.float().unsqueeze(1)

    return prototypes


def calculate_nearest_labeled_neighbors(features: torch.tensor, model: torch.nn.Module, k: int, labels: torch.tensor,
                                        num_classes: int, key: str = "logits_lab") -> torch.tensor:
    prototypes = calculate_prototypes(features, labels, num_classes)
    prototype_features = model.forward_heads(prototypes.float())
    nearest_neighbors = prototype_features[key].topk(k=k, dim=1).indices

    return nearest_neighbors
