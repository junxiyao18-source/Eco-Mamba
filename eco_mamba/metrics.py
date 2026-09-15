"""Action-level and ethogram-level evaluation metrics for Eco-Mamba."""

from typing import Dict, Sequence

import numpy as np
from sklearn.metrics import average_precision_score


def _sigmoid(logits: np.ndarray) -> np.ndarray:
    return np.where(logits >= 0, 1 / (1 + np.exp(-logits)), np.exp(logits) / (1 + np.exp(logits)))


def build_group_targets(action_targets: np.ndarray, action_to_group: Dict[int, int], num_groups: int = 16) -> np.ndarray:
    group_targets = np.zeros((action_targets.shape[0], num_groups), dtype=action_targets.dtype)
    for action_id, group_id in action_to_group.items():
        group_targets[:, group_id] = np.maximum(group_targets[:, group_id], action_targets[:, action_id])
    return group_targets


def hierarchical_consistency_rate(action_probabilities, group_probabilities, action_to_group, threshold: float = .5) -> float:
    parent_ids = np.array([action_to_group[action_id] for action_id in range(action_probabilities.shape[1])])
    action_active = action_probabilities > threshold
    parent_active = group_probabilities[:, parent_ids] > threshold
    return float((action_active == parent_active).mean())


def compute_group_map(action_logits, action_targets, action_to_group, group_names: Sequence[str]):
    probabilities = _sigmoid(action_logits)
    group_targets = build_group_targets(action_targets, action_to_group, len(group_names))
    group_probabilities = np.zeros_like(group_targets, dtype=np.float32)
    for action_id, group_id in action_to_group.items():
        group_probabilities[:, group_id] = np.maximum(group_probabilities[:, group_id], probabilities[:, action_id])
    group_ap = {}
    for group_id, group_name in enumerate(group_names):
        try:
            group_ap[group_name] = float(average_precision_score(group_targets[:, group_id], group_probabilities[:, group_id]))
        except ValueError:
            group_ap[group_name] = 0.0
    return float(np.mean(list(group_ap.values()))), group_ap


def collect_per_class_auc(action_logits, action_targets, action_names: Sequence[str]):
    probabilities = _sigmoid(action_logits)
    results = {}
    for action_id, action_name in enumerate(action_names):
        try:
            results[action_name] = float(average_precision_score(action_targets[:, action_id], probabilities[:, action_id]))
        except ValueError:
            results[action_name] = 0.0
    return results
