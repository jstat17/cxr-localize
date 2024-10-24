import torch as th
from torch import nn
from torch.nn import DataParallel
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm
from pathlib import Path
import os
import skimage

from vision.segment_anything.medsam import SAM_NORMALIZATION_DICT
from utils import transform


class PaddedImageDataset(Dataset):
    def __init__(self, images_path: Path, set_shape: tuple[int, int], norm_dict: dict[str, list] = SAM_NORMALIZATION_DICT) -> None:
        self.images_path = images_path
        self.filenames = os.listdir(self.images_path)
        self.set_shape = set_shape
        self.norm_dict = norm_dict

    def _pad_image_square(self, image: np.ndarray) -> np.ndarray:
        channels, rows, cols = image.shape
        if rows != cols:
            max_dim = max(rows, cols)
            image_padded = np.zeros((channels, max_dim, max_dim), dtype=image.dtype)
            image_padded[:, 0:rows, 0:cols] = image
            image = image_padded

        return image
    
    def _resize_image(self, image: np.ndarray) -> np.ndarray:
        channels = image.shape[0]
        image_resized = np.zeros((channels, *self.set_shape), dtype=image.dtype)
        for i in range(channels):
            image_resized[i] = transform.resize(image[i], self.set_shape)

        return image_resized

    def _load_image(self, image_path: Path) -> np.ndarray:
        # load image
        image = skimage.io.imread(image_path)

        # convert to float
        image = image.astype(np.float32)

        # ensure image is grayscale (single channel)
        if len(image.shape) == 3:
            image = image[:, :, 0] # (H, W, C) -> (H, W)

        # copy to 3 channel dimensions
        image_3_channel = np.zeros(
            shape = (3, *image.shape),
            dtype = np.float32
        )
        for i in range(3):
            image_3_channel[i] = image
        
        image = image_3_channel # (3, H, W)

        # normalize
        mean = self.norm_dict['mean']
        mean = np.array(mean, dtype=np.float32)
        mean = mean[:, np.newaxis, np.newaxis] # (3, 1, 1)

        std = self.norm_dict['std']
        std = np.array(std, dtype=np.float32)
        std = std[:, np.newaxis, np.newaxis] # (3, 1, 1)

        image = np.subtract(
            image,
            mean,
        )
        image = np.divide(
            image,
            std
        )

        # pad and resize
        image = self._pad_image_square(image)
        image = self._resize_image(image)

        return image
    
    def __len__(self) -> int:
        return len(self.filenames)
    
    def __getitem__(self, idx: int) -> tuple[th.Tensor, th.Tensor]:
        # load image from disk
        filename = self.filenames[idx]
        image_path = self.images_path / filename
        image = self._load_image(image_path)

        # convert to PyTorch tensor
        image = th.from_numpy(image)

        return image, filename


def get_embeddings(model: nn.Module, dataloader: DataLoader, device: str, parallel: bool) -> dict[str, np.ndarray]:
    if parallel:
        model = DataParallel(model)
    elif device == "cuda":
        device += ":0"
    
    model = model.to(device)
    model.eval()
    pbar = tqdm(dataloader, desc="Computing embeddings", unit="batch")

    embeddings_dict = dict()
    for batch in pbar:
        images, filenames = batch
        images = images.to(device)
        with th.no_grad():
            embeddings = model(images)

        embeddings = embeddings.detach().cpu().numpy()
        for embedding, filename in zip(embeddings, filenames):
            embeddings_dict[filename] = embedding

        # clear out intermediate results
        th.cuda.empty_cache()

    return embeddings_dict