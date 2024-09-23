from argparse import ArgumentParser
from pathlib import Path
import os
from tqdm import tqdm
import skimage

from utils import transform

"""Resize cropped images.
   Run from cxr-localize:
        python -m script.resize_images /path/to/crop/root size
"""

def main(crop_path: Path, size: int) -> None:
    # make directory to move images to
    images_path = crop_path / "images"
    images_resize_path = crop_path / f"images-{size}"
    os.makedirs(images_resize_path, exist_ok=True)

    # resize and save images with progress bar
    n_saved = 0
    all_files = os.listdir(images_path)
    pbar = tqdm(
        total = len(all_files),
        desc = "Moving files..."
    )
    new_shape = (size, size)
    for file in all_files:
        # load image and resize
        image_old_path = images_path / file
        image_new_path = images_resize_path / file

        image = skimage.io.imread(image_old_path)
        image = transform.resize(image, new_shape)

        # save resized image
        skimage.io.imsave(image_new_path, image)
        n_saved += 1

        pbar.update(1)

    pbar.close()
    print(f"Resized {n_saved} file/s and saved to {images_resize_path}")


if __name__ == "__main__":
    # set up argument parsing
    parser = ArgumentParser(description='Resize cropped images')
    parser.add_argument('crop_path', type=str, help="Path to the cropped images root (parent directory that includes logs, images and masks)")
    parser.add_argument('size', type=float, help="Image will be resized to (size, size) shape")

    # parse command line arguments
    args = parser.parse_args()
    crop_path = Path(args.crop_path)
    size = args.intersection

    main(crop_path, size)