import torch as th
from torch import nn
from torchvision.models import convnext_base, ConvNeXt_Base_Weights

def get_convnext_base(num_classes: int, weights: str | None) -> nn.Module:
    if isinstance(weights, str):
        weights = weights.casefold()
    
    match weights:
        # load imagenet 1k weights
        case "imagenet":
            loaded_weights = ConvNeXt_Base_Weights.IMAGENET1K_V1

        # use random weights
        case None:
            loaded_weights = None

    model = ConvNeXt_Base(
        num_classes = num_classes,
        weights = loaded_weights
    )

    return model

class ConvNeXt_Base(nn.Module):
    def __init__(self, num_classes: int, weights: dict | None) -> None:
        super().__init__()
        self.model = convnext_base(
            weights = weights
        )
        self.model.classifier = nn.Sequential(
            nn.LayerNorm(self.model.classifier[1].in_features),
            nn.Linear(self.model.classifier[1].in_features, num_classes),
            nn.Sigmoid()
        )

    def forward(self, x: th.Tensor) -> th.Tensor:
        x = self.model(x)
        return x