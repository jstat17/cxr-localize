import torch as th
from torch import nn
import timm

class ViT_S(nn.Module):
    model_info: str = "vit_small_patch16_224"

    def __init__(self, num_classes: int, weights: str | None) -> None:
        super().__init__()
        self.fullname = "ViT-S"

        if weights is not None:
            timm_model = self.model_info + weights
            pretrained = True
        else:
            timm_model = self.model_info
            pretrained = False
        
        self.model = timm.create_model(
            timm_model,
            pretrained = pretrained,
            num_classes = num_classes
        )

    def forward(self, x: th.Tensor) -> th.Tensor:
        x = self.model(x)
        return x


def get_vit_s(num_classes: int, weights: str | None) -> ViT_S:
    if isinstance(weights, str):
        weights = weights.casefold()
    
    match weights:
        # load imagenet 1k weights
        case "imagenet":
            loaded_weights = ".augreg_in1k"

        # use random weights
        case None:
            loaded_weights = None

    model = ViT_S(
        num_classes = num_classes,
        weights = loaded_weights
    )

    return model
