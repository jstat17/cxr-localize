import torch as th
from torch import nn
from torchvision.models import swin_s, Swin_S_Weights

def get_swin_s(num_classes: int, weights: str | None) -> nn.Module:
    if isinstance(weights, str):
        weights = weights.casefold()
    
    match weights:
        # load imagenet 1k weights
        case "imagenet":
            loaded_weights = Swin_S_Weights.IMAGENET1K_V1

        # use random weights
        case None:
            loaded_weights = None

    model = Swin_S(
        num_classes = num_classes,
        weights = loaded_weights
    )

    return model

class Swin_S(nn.Module):
    def __init__(self, num_classes: int, weights: dict | None) -> None:
        super().__init__()
        self.model = swin_s(
            weights = weights
        )
        self.model.head = nn.Sequential(
            nn.Linear(self.model.head.in_features, num_classes),
            nn.Sigmoid()
        )

    def forward(self, x: th.Tensor) -> th.Tensor:
        x = self.model(x)
        return x
