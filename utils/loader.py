from pathlib import Path
import numpy as np
import os
import numpy as np
import torch as th
import skimage
import torchxrayvision as xrv
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from torchvision.models import ResNet50_Weights
import pandas as pd

from utils import transform

class XRVDataset(Dataset):
    """Dataset for TorchXRayVision models, which loads images from a path, resizes them to 512 x 512,\
    normalizes to the [-1024, 1024] range as float32
    """

    set_shape: tuple[int, int] = (512, 512)
    
    def __init__(self, images_path: Path, save_path_images: Path | None = None) -> None:
        self.images_path = images_path
        self.filenames = os.listdir(self.images_path)

        # remove images that have already been cropped previously
        if save_path_images is not None:
            filenames_existing_set = set(os.listdir(save_path_images))
            filenames_set = set(self.filenames)
            self.filenames = list(
                filenames_set - filenames_existing_set
            )

    def __len__(self) -> int:
        return len(self.filenames)

    def __getitem__(self, idx: int) -> tuple[th.Tensor, th.Tensor, th.Tensor]:
        try:
            # load image
            image_path = self.images_path / Path(self.filenames[idx])
            image = skimage.io.imread(image_path)
            original_size = list(image.shape) # (height, width)

            # resize to 512 x 512
            image = transform.resize(image, self.set_shape)
            image = image[np.newaxis, :] # add channel dimension (1, H, W)

            # normalize to [-1024, 1024] and convert to float
            max_dtype_value = transform.get_max_value(image)
            image = xrv.datasets.normalize(image, max_dtype_value)

            # convert to PyTorch tensor
            tensor = th.from_numpy(image).float()

            return tensor, th.tensor(original_size), self.filenames[idx]
        
        except OSError as e:
            print(f"OSError in loading {image_path} : {e}")
            return None, None, None
        

class MulticlassDataset(Dataset):

    def __init__(self, df: pd.DataFrame, images_path: str, img_shape: tuple[int, int], split: str, hash_percentile: float, possible_labels: list[str]) -> None:
        self.df = df.copy(deep=False)
        self.images_path = images_path
        self.img_shape = img_shape
        self.split = split.casefold()
        self.possible_labels = possible_labels

        self.hash_percentile = hash_percentile
        self.possible_hashes = 2**32
        self.hash_value_split = int(self.possible_hashes * self.hash_percentile)

        self.filenames = self._get_filenames()
        self.labels = self._get_labels()

    def _hash_filename(self, filename: str) -> int:
        """Generate a 32-bit FNV-1a hash value for a given filename."""
        return transform.fnv1a_32(filename)
    
    def _get_filenames(self) -> list[str]:
        all_filenames = os.listdir(self.images_path)
        selected_filenames = []
        for filename in all_filenames:
            hash = self._hash_filename(filename)
            if (
                (hash <= self.hash_value_split and self.split == "train")
                or
                (hash > self.hash_value_split and self.split == "test")
            ):
                selected_filenames.append(filename)

        return selected_filenames
    
    def _get_labels(self) -> list[np.ndarray]:
        labels = []
        for filename in self.filenames:
            image_labels = self.df[
                self.df['Filename'] == filename
            ]['Labels'].values

            # if the labels are not an empty list
            if image_labels:
                image_labels = image_labels[0]

            vec = transform.encode_multiclass_one_hot(
                possible_labels = self.possible_labels,
                labels = image_labels
            )
            labels.append(vec)

        return labels
    
    def __len__(self) -> int:
        return len(self.filenames)
    
    def __getitem__(self, idx: int) -> tuple[th.Tensor, th.Tensor, th.Tensor]:
        # load image
        image_path = self.images_path / Path(self.filenames[idx])
        image = skimage.io.imread(image_path)

        # convert to float and normalize to [0., 1.]
        max_dtype_value = transform.get_max_value(image)
        image = image.astype(np.float32)
        image = image / max_dtype_value

        # resize
        image = transform.resize(image, self.img_shape)

        # copy to 3 channel dimensions
        image = np.stack([image, image, image], axis=0) # (3, H, W)

        # get image labels
        labels = self.labels[idx]

        # convert to PyTorch tensor
        tensor_image = th.from_numpy(image)
        tensor_labels = th.from_numpy(labels)

        return tensor_image, tensor_labels