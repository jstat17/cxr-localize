import torch as th
from torch import nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from torchvision.models import efficientnet_b4, EfficientNet_B4_Weights

from vision.vision_utils import linearize_state_dict

class EfficientNet_B0(nn.Module):
    def __init__(self, num_classes: int, weights: EfficientNet_B0_Weights | None) -> None:
        super().__init__()
        self.fullname = "EfficientNet-B0"
        self.model = efficientnet_b0(
            weights = weights
        )
        self.model.classifier = nn.Sequential(
            self.model.classifier[0],  # Dropout layer
            nn.Linear(self.model.classifier[1].in_features, num_classes)
        )

    def forward(self, x: th.Tensor) -> th.Tensor:
        x = self.model(x)
        return x
    
class EfficientNet_B4(nn.Module):
    def __init__(self, num_classes: int, weights: EfficientNet_B4_Weights | None) -> None:
        super().__init__()
        self.fullname = "EfficientNet-B4"
        self.model = efficientnet_b4(
            weights = weights
        )
        self.model.classifier = nn.Sequential(
            self.model.classifier[0],  # Dropout layer
            nn.Linear(self.model.classifier[1].in_features, num_classes)
        )

    def forward(self, x: th.Tensor) -> th.Tensor:
        x = self.model(x)
        return x


def get_efficientnet_b0(num_classes: int, weights: str | dict | None = None) -> EfficientNet_B0:
    # select pretrained weights
    match weights:
        case str():
            pretrained_weights = weights.casefold()

        case dict() | None | _:
            pretrained_weights = None
    
    match pretrained_weights:
        # load imagenet 1k weights
        case "imagenet":
            loaded_weights = EfficientNet_B0_Weights.IMAGENET1K_V1

        # use random weights
        case None:
            loaded_weights = None

    model = EfficientNet_B0(
        num_classes = num_classes,
        weights = loaded_weights
    )

    # load state dict into model
    if isinstance(weights, dict):
        state_dict = linearize_state_dict(weights)
        model.load_state_dict(state_dict)

    return model

def get_efficientnet_b4(num_classes: int, weights: str | dict | None = None) -> EfficientNet_B4:
    # select pretrained weights
    match weights:
        case str():
            pretrained_weights = weights.casefold()

        case dict() | None | _:
            pretrained_weights = None
    
    match pretrained_weights:
        # load imagenet 1k weights
        case "imagenet":
            loaded_weights = EfficientNet_B4_Weights.IMAGENET1K_V1

        # use random weights
        case None:
            loaded_weights = None

    model = EfficientNet_B4(
        num_classes = num_classes,
        weights = loaded_weights
    )

    # load state dict into model
    if isinstance(weights, dict):
        state_dict = linearize_state_dict(weights)
        model.load_state_dict(state_dict)

    return model