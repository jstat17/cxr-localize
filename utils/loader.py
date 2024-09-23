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

PYTORCH_SECOND_NORMALIZATION_DICT = {
    'mean': [0.485, 0.456, 0.406],
    'std': [0.229, 0.224, 0.225]
}

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

    def __init__(self, df: pd.DataFrame, images_path: str, img_shape: int | tuple[int, int], split: str, hash_percentile: float, possible_labels: list[str], second_norm_dict: dict[str, list] | None = PYTORCH_SECOND_NORMALIZATION_DICT) -> None:
        self.df = df.copy(deep=False)
        self.images_path = images_path

        if isinstance(img_shape, int):
            self.img_shape = (img_shape, img_shape)
        else:
            self.img_shape = img_shape

        self.split = split.casefold()
        self.possible_labels = possible_labels

        self.hash_percentile = hash_percentile
        self.possible_hashes = 2**32
        self.hash_value_split = int(self.possible_hashes * self.hash_percentile)

        self.filenames = self._get_filenames()
        self.labels = th.from_numpy(
            self._get_labels()
        )

        self.second_norm_dict = second_norm_dict

    def _hash_filename(self, filename: str) -> int:
        """Generate a 32-bit FNV-1a hash value for a given filename.

        Args:
            filename (str): Filename as string

        Returns:
            int: 32-bit hash value
        """
        return transform.fnv1a_32(filename)
    
    def _get_filenames(self) -> list[str]:
        """Get a list of filenames from the specified images_path that fall in the split

        Returns:
            list[str]: List of string filenames
        """
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
    
    def _get_labels(self) -> np.ndarray:
        """Get an array of multi-class labels for all images that fall in the split

        Returns:
            np.ndarray: List of multi-hot numpy arrays, shape (num_images, num_classes)
        """
        n = len(self)
        labels = np.zeros(
            shape = (len(self), len(self.possible_labels)),
            dtype = np.float32
        )

        for idx in range(n):
            filename = self.filenames[idx]
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
            labels[idx] = vec

        return labels
    
    def _load_normalize_image(self, image_path: Path) -> np.ndarray:
        """Load image from image_path and normalize it

        Args:
            image_path (Path): Path to the loaded image

        Returns:
            np.ndarray: 3-channel normalized image, shape (3, H, W)
        """
        # load image
        image = skimage.io.imread(image_path)

        # convert to float and normalize to [0., 1.]
        max_dtype_value = transform.get_max_value(image)
        image = image.astype(np.float32)
        image = image / max_dtype_value

        # copy to 3 channel dimensions
        image_3_channel = np.zeros(
            shape = (3, *image.shape),
            dtype = np.float32
        )
        for i in range(3):
            image_3_channel[i] = image
        
        image = image_3_channel # (3, H, W)

        # do second normalization
        if self.second_norm_dict is not None:
            mean = self.second_norm_dict['mean']
            mean = np.array(mean, dtype=np.float32)
            mean = mean[:, np.newaxis, np.newaxis] # (3, 1, 1)

            std = self.second_norm_dict['std']
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

        return image

    def __len__(self) -> int:
        return len(self.filenames)
    
    def __getitem__(self, idx: int) -> tuple[th.Tensor, th.Tensor]:
        # load image from disk
        image_path = self.images_path / self.filenames[idx]
        image = self._load_normalize_image(image_path)

        # resize
        image = transform.resize(image, self.img_shape)

        # convert to PyTorch tensor
        image = th.from_numpy(image)

        # get labels from memory
        labels = self.labels[idx]

        return image, labels
    

class MulticlassDatasetInMemory(MulticlassDataset):

    def __init__(self, df: pd.DataFrame, images_path: str, img_shape: int | tuple[int, int], split: str, hash_percentile: float, possible_labels: list[str], second_norm_dict: dict[str, list] | None = PYTORCH_SECOND_NORMALIZATION_DICT) -> None:
        super().__init__(df, images_path, img_shape, split, hash_percentile, possible_labels, second_norm_dict)
        
        self.images = th.from_numpy(
            self._load_normalize_all_images()
        )

    def _load_normalize_all_images(self) -> np.ndarray:
        """Load all images into memory and normalize them

        Returns:
            np.ndarray: Array of images, shape (num_images, 3, H, W)
        """
        n = len(self)
        images = np.zeros(
            shape = (n, 3, *self.img_shape),
            dtype = np.float32
        )

        for idx in range(n):
            image_path = self.images_path / self.filenames[idx]
            image = self._load_normalize_image(image_path)

            images[idx] = image

        return images
    
    def __getitem__(self, idx: int) -> tuple[th.Tensor, th.Tensor]:
        # get image from memory
        image = self.images[idx]

        # get labels from memory
        labels = self.labels[idx]

        return image, labels