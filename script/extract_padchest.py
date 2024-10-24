from argparse import ArgumentParser
from pathlib import Path
import numpy as np
from natsort import natsorted

from dataset import padchest

"""Extract specified files from PadChest.
   Run from cxr-localize:
        python -m script.extract_padchest /path/to/PadChest /path/to/extracted ..args..
"""

def main(padchest_path: Path, extract_path: Path, projections_kept: list[str] | None, labels_kept: list[str] | None) -> None:
    # get dataframe of file info
    df = padchest.get_padchest_dataframe(
        padchest_path = padchest_path,
        projections_kept = projections_kept,
        labels_kept = labels_kept
    )

    # create a dictionary from zip number to list of filenames
    zip_nums = np.sort(df['Zip Number'].unique())
    zip_to_filename = dict()
    for zip_num in zip_nums:
        filenames = df[
            df['Zip Number'] == zip_num
        ]['Filename']
        zip_to_filename[zip_num] = natsorted(filenames.to_list())

    # extract specified PadChest files to the extract path
    padchest.extract_files_from_zips(
        zip_to_filenames = zip_to_filename,
        padchest_path = padchest_path,
        extract_path = extract_path
    )

if __name__ == "__main__":
    # set up argument parsing
    parser = ArgumentParser(description='Extract specified files from PadChest')
    parser.add_argument('padchest_path', type=str, help='Path to originally-downloaded PadChest dataset')
    parser.add_argument('extract_path', type=str, help="Desired extract path for all files")
    parser.add_argument('--projections_kept', type=str, default="AP", help="Projections to keep")
    parser.add_argument('--labels_kept', type=str, default="Shenzhen", help="Labels to keep")

    # parse command line arguments
    args = parser.parse_args()
    padchest_path = Path(args.padchest_path)
    extract_path = Path(args.extract_path)

    projections_kept = args.projections_kept.casefold()
    match projections_kept:
        case "none":
            projections_kept = None
        case "ap":
            projections_kept = padchest.PROJECTIONS_KEPT

    labels_kept = args.labels_kept.casefold()
    match labels_kept:
        case "none":
            labels_kept = None
        case "shenzhen":
            labels_kept = padchest.PADCHEST_ABNORMALITIES_COMMON_SHENZHEN

    main(
        padchest_path = padchest_path,
        extract_path = extract_path,
        projections_kept = projections_kept,
        labels_kept = labels_kept
    )