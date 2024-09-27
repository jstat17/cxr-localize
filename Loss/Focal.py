import torch as th
import torch.nn as nn

class FocalLoss(nn.Module):
    def __init__(self, alpha: float = 1.0, gamma: float = 2.0, reduction: str = 'mean') -> None:
        """
        Focal Loss for multi-label classification.

        Args:
            alpha (float): Balancing factor for positive/negative examples.
            gamma (float): Focusing parameter to adjust the rate at which easy examples are down-weighted.
            reduction (str): Specifies the reduction to apply to the output: 'none' | 'mean' | 'sum'.
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs: th.Tensor, targets: th.Tensor) -> th.Tensor:
        """
        Compute the focal loss.

        Args:
            inputs (torch.Tensor): Predicted logits (raw scores) from the model.
            targets (torch.Tensor): Ground truth binary labels (0 or 1) for each class.

        Returns:
            torch.Tensor: Computed focal loss value.
        """
        # Apply sigmoid to get probabilities
        BCE_loss = nn.BCEWithLogitsLoss(reduction='none')(inputs, targets)
        
        # Compute p_t (probability of the true class)
        pt = th.where(targets == 1, th.sigmoid(inputs), 1 - th.sigmoid(inputs))

        # Focal loss calculation
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss
        
        # Apply reduction
        if self.reduction == 'mean':
            return F_loss.mean()
        elif self.reduction == 'sum':
            return F_loss.sum()
        return F_loss
