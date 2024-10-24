import torch as th
from torch import nn
from functools import partial
from pathlib import Path

from .image_encoder import ImageEncoderViT

SAM_NORMALIZATION_DICT = {
    'mean': [123.675, 116.28, 103.53],
    'std': [58.395, 57.12, 57.375]
}
CHECKPOINT_PATH = Path.home() / "medsam-weights" / "medsam_vit_b.pth"


class MedSAM_Encoder(nn.Module):
    def __init__(self, checkpoint_path: Path) -> None:
        super().__init__()
        self.full_name = "MedSAM-Image-Encoder_ViT-B"

        # set ViT B params
        encoder_embed_dim = 768
        encoder_depth = 12
        encoder_num_heads = 12
        encoder_global_attn_indexes = [2, 5, 8, 11]
        prompt_embed_dim = 256
        image_size = 1024
        vit_patch_size = 16

        self.image_encoder = ImageEncoderViT(
            depth = encoder_depth,
            embed_dim = encoder_embed_dim,
            img_size = image_size,
            mlp_ratio = 4,
            norm_layer = partial(th.nn.LayerNorm, eps=1e-6),
            num_heads = encoder_num_heads,
            patch_size = vit_patch_size,
            qkv_bias = True,
            use_rel_pos = True,
            global_attn_indexes = encoder_global_attn_indexes,
            window_size = 14,
            out_chans = prompt_embed_dim,
        )

        # load weights
        checkpoint = th.load(checkpoint_path, weights_only=True)
        # filter out the weights that belong to the image encoder
        checkpoint = {k: v for k, v in checkpoint.items() if k.startswith('image_encoder')}
        
        self.load_state_dict(checkpoint)

        # pooling layer to reduce dimensionality of 256-channel feature map
        self.pool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )

    def forward(self, x: th.Tensor) -> th.Tensor:
        x = self.image_encoder(x)
        x = self.pool(x)

        return x


def get_medsam_image_encoder():
    model = MedSAM_Encoder(
        checkpoint_path = CHECKPOINT_PATH
    )
    
    return model