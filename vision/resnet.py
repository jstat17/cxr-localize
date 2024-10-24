import torch as th
from torch import nn
from torchvision.models import resnet50, ResNet50_Weights

from vision.vision_utils import linearize_state_dict


class ResNet50(nn.Module):
    def __init__(self, num_classes: int, weights: ResNet50_Weights | None) -> None:
        super().__init__()
        self.fullname = "ResNet50"
        self.model = resnet50(
            weights = weights
        )
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x: th.Tensor) -> th.Tensor:
        x = self.model(x)
        return x


def get_resnet50(num_classes: int, weights: str | dict | None = None) -> ResNet50:
    # select pretrained weights
    match weights:
        case str():
            pretrained_weights = weights.casefold()

        case dict() | None | _:
            pretrained_weights = None
    
    match pretrained_weights:
        # load imagenet 1k weights
        case "imagenet":
            loaded_weights = ResNet50_Weights.IMAGENET1K_V2

        # use random weights
        case None:
            loaded_weights = None

    model = ResNet50(
        num_classes = num_classes,
        weights = loaded_weights
    )

    # load state dict into model
    if isinstance(weights, dict):
        state_dict = linearize_state_dict(weights)
        model.load_state_dict(state_dict)

    return model