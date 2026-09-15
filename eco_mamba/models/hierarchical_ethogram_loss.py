"""Losses for the Eco-Mamba hierarchical ethogram decoder."""

from typing import Dict

import torch
from torch import nn
import torch.nn.functional as F


class EthogramConsistencyLoss(nn.Module):
    """Penalize fine-grained action probabilities above their parent group."""

    def __init__(self, action_to_group: Dict[int, int]):
        super().__init__()
        parent_ids = [action_to_group[action_id] for action_id in range(len(action_to_group))]
        self.register_buffer("parent_ids", torch.tensor(parent_ids, dtype=torch.long), persistent=False)

    def forward(self, action_logits: torch.Tensor, group_logits: torch.Tensor) -> torch.Tensor:
        action_probabilities = torch.sigmoid(action_logits)
        group_probabilities = torch.sigmoid(group_logits)
        # Eq. (19) in the paper.
        return F.relu(action_probabilities - group_probabilities[:, self.parent_ids]).mean()


class HierarchicalEthogramLoss(nn.Module):
    """Joint action BCE, group BCE, and ethogram-consistency objective."""

    def __init__(
        self,
        action_to_group: Dict[int, int],
        num_groups: int = 16,
        action_weight: float = 1.0,
        group_weight: float = 1.0,
        consistency_weight: float = 0.2,
    ):
        super().__init__()
        self.action_weight = action_weight
        self.group_weight = group_weight
        self.consistency_weight = consistency_weight
        self.action_bce = nn.BCEWithLogitsLoss()
        self.group_bce = nn.BCEWithLogitsLoss()
        self.consistency_loss = EthogramConsistencyLoss(action_to_group)

    def forward(self, action_logits, action_targets, group_logits, group_targets):
        action_loss = self.action_bce(action_logits, action_targets)
        group_loss = self.group_bce(group_logits, group_targets)
        hierarchy_loss = self.consistency_loss(action_logits, group_logits)
        total_loss = self.action_weight * action_loss + self.group_weight * group_loss + self.consistency_weight * hierarchy_loss
        return total_loss, {
            "action_loss": action_loss.item(),
            "group_loss": group_loss.item(),
            "hierarchy_loss": hierarchy_loss.item(),
        }
