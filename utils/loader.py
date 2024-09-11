from pathlib import Path
import numpy as np
import os
import numpy as np
import torch as th
import skimage
import torchxrayvision as xrv
from torch.utils.data import Dataset

from utils import transform

class XRVDataset(Dataset):
    """Dataset for TorchXRayVision models, which loads images from a path, resizes them to 512 x 512,\
    normalizes to the [-1024, 1024] range as float32
    """

    set_shape: tuple[int, int] = (512, 512)
    
    def __init__(self, images_path: Path):
        self.images_path = images_path
        self.filenames = os.listdir(self.images_path)

    def __len__(self) -> int:
        return len(self.filenames)

    def __getitem__(self, idx: int) -> tuple[th.Tensor, th.Tensor, th.Tensor]:
        # load image
        image_path = self.images_path / Path(self.filenames[idx])
        image = skimage.io.imread(image_path)
        original_size = list(image.shape) # (height, width)

        # resize to 512 x 512
        image = transform.resize(image, self.set_shape)
        image = image[np.newaxis, :] # add channel dimension (1, H, W)

        # normalize to [-1024, 1024] and convert to float
        image = xrv.datasets.normalize(image, np.iinfo(image.dtype).max)

        # convert to PyTorch tensor
        tensor = th.from_numpy(image).float()

        return tensor, th.tensor(original_size), self.filenames[idx]