import torch as th
from torch import nn
from torchvision.models import resnet50, ResNet50_Weights


class ResNet50(nn.Module):
    def __init__(self, num_classes: int, weights: dict | None) -> None:
        super().__init__()
        self.fullname = "ResNet50"
        self.model = resnet50(
            weights = weights
        )
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x: th.Tensor) -> th.Tensor:
        x = self.model(x)
        return x


def get_resnet50(num_classes: int, weights: str | None) -> ResNet50:
    if isinstance(weights, str):
        weights = weights.casefold()
    
    match weights:
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

    return model