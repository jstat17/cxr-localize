from argparse import ArgumentParser
from pathlib import Path
import os
from tqdm import tqdm
import skimage

from utils import transform

"""Resize images.
   Run from cxr-localize:
        python -m script.resize_images /path/to/images /path/to/resized size
"""

def main(images_path: Path, images_resize_path: Path, size: int) -> None:
    # make directory to save resized images to
    os.makedirs(images_resize_path, exist_ok=True)

    # resize and save images with progress bar
    n_saved = 0
    all_files = os.listdir(images_path)
    pbar = tqdm(
        total = len(all_files),
        desc = "Resizing and saving files..."
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
    parser = ArgumentParser(description='Resize images')
    parser.add_argument('images_path', type=str, help="Path to images")
    parser.add_argument('images_resize_path', type=str, help="Desired path for resized images")
    parser.add_argument('size', type=int, help="Image will be resized to (size, size) shape")

    # parse command line arguments
    args = parser.parse_args()
    images_path = Path(args.images_path)
    images_resize_path = Path(args.images_resize_path)
    size = args.size

    main(images_path, images_resize_path, size)