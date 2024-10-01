import torch as th
from torch import nn
from torchvision.models import convnext_base, ConvNeXt_Base_Weights
from torchvision.models import convnext_small, ConvNeXt_Small_Weights


class ConvNeXt_B(nn.Module):
    def __init__(self, num_classes: int, weights: dict | None) -> None:
        super().__init__()
        self.fullname = "ConvNeXt-B"
        self.model = convnext_base(
            weights = weights
        )
        self.model.classifier = nn.Sequential(
            self.model.classifier[0],  # Flatten layer
            self.model.classifier[1],  # LayerNorm layer
            nn.Linear(self.model.classifier[2].in_features, num_classes)
        )

    def forward(self, x: th.Tensor) -> th.Tensor:
        x = self.model(x)
        return x
    
class ConvNeXt_S(nn.Module):
    def __init__(self, num_classes: int, weights: dict | None) -> None:
        super().__init__()
        self.fullname = "ConvNeXt-S"
        self.model = convnext_small(
            weights = weights
        )
        self.model.classifier = nn.Sequential(
            self.model.classifier[0],  # Flatten layer
            self.model.classifier[1],  # LayerNorm layer
            nn.Linear(self.model.classifier[2].in_features, num_classes)
        )

    def forward(self, x: th.Tensor) -> th.Tensor:
        x = self.model(x)
        return x


def get_convnext_b(num_classes: int, weights: str | None) -> ConvNeXt_B:
    if isinstance(weights, str):
        weights = weights.casefold()
    
    match weights:
        # load imagenet 1k weights
        case "imagenet":
            loaded_weights = ConvNeXt_Base_Weights.IMAGENET1K_V1

        # use random weights
        case None:
            loaded_weights = None

    model = ConvNeXt_B(
        num_classes = num_classes,
        weights = loaded_weights
    )

    return model

def get_convnext_s(num_classes: int, weights: str | None) -> ConvNeXt_B:
    if isinstance(weights, str):
        weights = weights.casefold()
    
    match weights:
        # load imagenet 1k weights
        case "imagenet":
            loaded_weights = ConvNeXt_Small_Weights.IMAGENET1K_V1

        # use random weights
        case None:
            loaded_weights = None

    model = ConvNeXt_S(
        num_classes = num_classes,
        weights = loaded_weights
    )

    return model