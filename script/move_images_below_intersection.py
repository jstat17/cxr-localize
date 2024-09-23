from argparse import ArgumentParser
from pathlib import Path
import os
import shutil
from tqdm import tqdm

from utils import analyze

"""Move crops to `images-unused` folder if they fall below a set intersection percentage.
   Run from cxr-localize:
        python -m script.move_images_below_intersection /path/to/PadChest /path/to/crop/root intersection_val
"""

def main(padchest_path: Path, crop_path: Path, intersection: float) -> None:
    # get filenames to move
    filename_to_crop_pct = analyze.get_cropped_below_intersection_pct(
        padchest_path = padchest_path,
        crop_path = crop_path,
        intersection = intersection
    )
    filenames_to_move = [f"{filename}.png" for filename in filename_to_crop_pct.keys()]
    
    # make directory to move images to
    images_path = crop_path / "images"
    images_unused_path = crop_path / "images-unused"
    os.makedirs(images_unused_path, exist_ok=True)

    # move images with progress bar
    n_moved = 0
    all_files = os.listdir(images_path)
    pbar = tqdm(
        total = len(all_files),
        desc = "Moving files..."
    )
    for file in all_files:
        if file in filenames_to_move:
            image_old_path = images_path / file
            shutil.move(image_old_path, images_unused_path)
            n_moved += 1

        pbar.update(1)

    pbar.close()
    print(f"Moved {n_moved} file/s from {images_path} to {images_unused_path}")


if __name__ == "__main__":
    # set up argument parsing
    parser = ArgumentParser(description='Move cropped images that fall below the set intersection percenage')
    parser.add_argument('padchest_path', type=str, help='Path to originally-downloaded PadChest dataset')
    parser.add_argument('crop_path', type=str, help="Desired extract path for all files")
    parser.add_argument('intersection', type=float, help="Intersection percent limit below which to include")

    # parse command line arguments
    args = parser.parse_args()
    padchest_path = Path(args.padchest_path)
    crop_path = Path(args.crop_path)
    intersection = args.intersection

    main(padchest_path, crop_path, intersection)