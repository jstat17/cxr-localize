import torch as th
import torch.nn as nn
import torch.nn.functional as F

class WeightedBCEWithLogitsLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, inputs: th.Tensor, targets: th.Tensor) -> th.Tensor:
        # Calculate the weights for this batch
        pos_count = targets.sum()
        neg_count = targets.numel() - pos_count
        beta_p = (pos_count + neg_count) / pos_count
        beta_n = (pos_count + neg_count) / neg_count

        # Create a tensor of weights
        weights = th.where(targets == 1, beta_p, beta_n)

        # Calculate the loss
        loss = F.binary_cross_entropy_with_logits(inputs, targets, weight=weights, reduction='mean')
        return loss