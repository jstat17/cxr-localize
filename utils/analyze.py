import os
from pathlib import Path
import seaborn as sns
import matplotlib.pyplot as plt

from utils import dataset

def get_crop_intersection(padchest_path: Path, crop_path: Path) -> dict[str, float]:
    """Get the percentage intersection of cropped images with the original.

    Args:
        padchest_path (Path): Path to originally-downloaded PadChest dataset
        crop_path (Path): Path to the cropped images (parent directory that includes logs, images and masks)

    Returns:
        dict[str, float]: Dictionary from filename (without extension) to percentage of intersection
    """
    log_path = crop_path / "logs"
    df = dataset.get_padchest_dataframe(padchest_path)

    log_filenames = os.listdir(log_path)

    filename_to_crop_pct = dict()
    for log_file in log_filenames:
        with open(os.path.join(log_path, log_file), 'r') as f:
            d = eval(f.read())
            image_filename = log_file.split(".")[0] + ".png"
            
            image_rows = df[df['Filename'] == image_filename]['Rows'].values[0]
            image_cols = df[df['Filename'] == image_filename]['Columns'].values[0]
            image_area = image_rows * image_cols

            crop_area = abs(d['x2'] - d['x1']) * abs(d['y2'] - d['y1'])
            filename = log_file.split(".")[0]
            filename_to_crop_pct[filename] = crop_area / image_area

    return filename_to_crop_pct

def plot_bbox_image_intersection(padchest_path: Path, crop_path: Path) -> None:
    """Plot a histogram of the percentage intersection of cropped images.

    Args:
        padchest_path (Path): Path to originally-downloaded PadChest dataset
        crop_path (Path): Path to the cropped images (parent directory that includes logs, images and masks)
    """
    filename_to_crop_pcts = get_crop_intersection(padchest_path, crop_path)

    sns.histplot(filename_to_crop_pcts.values())
    plt.xlabel("Bounding Box Intersection with Image (%)")

    plt.show()

def get_cropped_below_intersection_pct(padchest_path: Path, crop_path: Path, intersection: float) -> dict[str, float]:
    """Get and filter the dictionary of percentage intersection of cropped images for cases below a set amount.

    Args:
        padchest_path (Path): Path to originally-downloaded PadChest dataset
        crop_path (Path): Path to the cropped images (parent directory that includes logs, images and masks)
        intersection (float): Intersection percent limit below which to include

    Returns:
        dict[str, float]: Dictionary from filename (without extension) to percentage of intersection
    """
    filename_to_crop_pcts = get_crop_intersection(padchest_path, crop_path)
    filtered_filename_to_crop_pct = {filename: pct for filename, pct in filename_to_crop_pcts.items() if pct <= intersection}

    return filtered_filename_to_crop_pct