import os
from pathlib import Path
import pandas as pd
import numpy as np
from natsort import natsorted
from ast import literal_eval
from zipfile import ZipFile
import hashlib
from collections import defaultdict
from collections.abc import Iterable

## Global constants
# columns to use for PadChest dataframe
SELECTED_COLUMNS = [
    'Name',
    'Zip Number',
    'Projection',
    'BitsStored_DICOM',
    'Rows_DICOM',
    'Columns_DICOM',
    'Labels',
    'Localizations',
    'StudyID',
    'PatientID',
    'ReportID',
]
RENAME_COLUMN_MAPPING = {
    'Name': 'Filename',
    'BitsStored_DICOM': 'Bits',
    'Rows_DICOM': 'Rows',
    'Columns_DICOM': 'Columns',
}
# X-ray projections to use
PROJECTIONS_KEPT = [
    "AP",
    "AP_horizontal",
    "PA"
]

# mapping between PadChest and Shenzhen dataset abnormalities
ABNORMALITY_MAPPING = [
    ("pleural effusion", "pleural effusion"),
    ("pleural effusion", "loculated pleural effusion"),
    ("pleural effusion", "loculated fissural effusion"),
    ("apical thickening", "apical pleural thickening"),
    ("single nodule (non-calcified)", "nodule"),
    ("single nodule (non-calcified)", "multiple nodules"),
    ("pleural thickening (non-apical)", "pleural thickening"),
    ("pleural thickening (non-apical)", "major fissure thickening"),
    ("pleural thickening (non-apical)", "minor fissure thickening"),
    ("calcified nodule", "calcified granuloma"),
    ("calcified nodule", "calcified adenopathy"),
    ("calcified nodule", "calcified densities"),
    ("small infiltrate (non-linear)", "infiltrates"),
    ("small infiltrate (non-linear)", "alveolar pattern"),
    ("cavity", "cavitation"),
    ("linear density", "fibrotic band"),
    ("linear density", "reticular interstitial pattern"),
    ("linear density", "interstitial pattern"),
    ("linear density", "reticulonodular interstitial pattern"),
    ("severe infiltrate (consolidation)", "consolidation"),
    ("thickening of the interlobar fissure", "fissure thickening"),
    ("thickening of the interlobar fissure", "major fissure thickening"),
    ("thickening of the interlobar fissure", "minor fissure thickening"),
    ("clustered nodule (2mm-5mm apart)", "multiple nodules"),
    ("moderate infiltrate (non-linear)", "infiltrates"),
    ("adenopathy", "adenopathy"),
    ("calcification (other than nodule and lymph node)", "calcified densities"),
    ("calcification (other than nodule and lymph node)", "calcified pleural plaques"),
    ("calcified lymph node", "calcified mediastinal adenopathy"),
    ("calcified lymph node", "calcified adenopathy"),
    ("miliary TB", "miliary opacities"),
    ("retraction", "volume loss"),
    ("retraction", "atelectasis"),
    ("retraction", "lobar atelectasis"),
    ("retraction", "segmental atelectasis"),
    ("retraction", "round atelectasis"),
    ("retraction", "laminar atelectasis"),
    ("retraction", "total atelectasis")
]
# dictionary from PadChest abnormality to list of Shenzhen abnormalities
PADCHEST_TO_SHENZHEN_MAPPING = defaultdict(list)
for abnorm_shen, abnorm_padchest in ABNORMALITY_MAPPING:
    PADCHEST_TO_SHENZHEN_MAPPING[abnorm_padchest].append(abnorm_shen)

# PadChest abnormalities that are common with Shenzhen
PADCHEST_ABNORMALITIES_COMMON_SHENZHEN = list(PADCHEST_TO_SHENZHEN_MAPPING.keys())

PADCHEST_DISEASES = [
    "pneumonia",
    "atypical pneumonia",
    "tuberculosis",
    "tuberculosis sequelae",
    "lung metastasis",
    "lymphangitis carcinomatosa",
    "lepidic adenocarcinoma",
    "pulmonary fibrosis",
    "post radiotherapy changes",
    "asbestosis signs",
    "emphysema",
    "COPD signs",
    "heart insufficiency",
    "respiratory distress",
    "pulmonary hypertension",
    "pulmonary artery hypertension",
    "pulmonary venous hypertension",
    "pulmonary edema",
    "bone metastasis"
]


## Data management functions
def get_padchest_dataframe(padchest_path: Path) -> pd.DataFrame:
    """Get the PadChest information dataframe of filenames, containing zip, labels etc.

    Args:
        padchest_path (Path): Path to originally-downloaded PadChest dataset

    Returns:
        pd.DataFrame: Dataframe of PadChest information
    """
    csv_path = padchest_path / "PADCHEST_chest_x_ray_images_labels_160K_01.02.19.csv.gz"
    df = pd.read_csv(
        csv_path,
        compression = 'gzip', # use gzip to decrompress
        index_col = 0 # use first column as index
    )

    # read in the unzip info from multiple text files
    fnames = natsorted(os.listdir(padchest_path))
    dfs = []
    for fname in fnames:
        if fname.endswith(".unzip-l.txt"):
            df_unzip = pd.read_csv(padchest_path / fname,
                skiprows = 3, # skip the first 3 rows
                skipfooter = 2, # skip the last 2 rows
                sep = r'\s{2,}', # use 2 or more whitespace characters as the delimiter
                engine = 'python', # use the python engine to handle the non-standard format
                names = ['Length', 'Datetime', 'Name'] # column names
            )
            df_unzip['Zip Number'] = int(fname.split(".")[0])
            dfs.append(df_unzip)

    df_unzip = pd.concat(dfs, axis=0).reset_index(drop=True)
    
    # merge unzip info with padchest info
    df = pd.merge(left=df_unzip, right=df, left_on='Name', right_on='ImageID')

    # select columns, and rename
    df = df[SELECTED_COLUMNS].rename(columns=RENAME_COLUMN_MAPPING)

    # drop rows with any NaN values
    df = df[~df.isna().any(axis=1)]

    # keep only specified projections
    df = df[
        df['Projection'].isin(PROJECTIONS_KEPT)
    ]

    # evaluate the string representations of the label and localization lists
    df['Labels'] = df['Labels'].apply(literal_eval)
    df['Localizations'] = df['Localizations'].apply(literal_eval)

    # remove label entries that contain a list of exactly one blank string
    df = df[~df['Labels'].apply(lambda x: x == [''])]

    # remove leading or trailing whitespace, and remove blank string labels
    strip_remove_blank = lambda labels: [label.strip() for label in labels if label != '']
    df['Labels'] = df['Labels'].apply(strip_remove_blank)
    df['Localizations'] = df['Localizations'].apply(strip_remove_blank)

    # keep only entries with abnormality labels that are common with Shenzhen
    df = df[
        df['Labels'].apply(
            lambda labels: intersect_iters(labels, PADCHEST_ABNORMALITIES_COMMON_SHENZHEN)
        )
    ]

    # reset index after all modifications
    df.reset_index(drop=True, inplace=True)

    return df

def extract_files_from_zips(zip_to_filenames: dict[int, list[str]], padchest_path: Path, extract_path: Path) -> None:
    """Extract files from multiple numbered PadChest zip files

    Args:
        zip_to_filenames (dict[int, list[str]]): Zip number to the list of filenames
        padchest_path (Path): Path to originally-downloaded PadChest dataset
        extract_path (Path): Desired extract path for all files
    """
    os.makedirs(extract_path, exist_ok=True)

    for zip_num, filenames in zip_to_filenames.items():
        zip_path = padchest_path / f"{zip_num}.zip"
        n_extracted = 0
        
        # read file contents of zip file without loading it into memory
        with ZipFile(zip_path, 'r') as z:
            for filename in z.namelist():
                # extract file from zip if it is in the list
                if filename in filenames:
                    z.extract(filename, path=extract_path)
                    n_extracted += 1

        print(f"Extracted {n_extracted} file/s to {zip_path}")

def verify_sha1sums(padchest_path: Path) -> None:
    """Verify the SHA-1 sums of the original PadChest files

    Args:
        padchest_path (Path): Path to originally-downloaded PadChest dataset
    """
    expected_sha1sums_filename = "sha1sums.txt"
    expected_sha1sums_path = padchest_path / expected_sha1sums_filename

    try:
        # read all lines and create a list of (expected_sha1, filename) tuples
        with open(expected_sha1sums_path, 'r') as f:
            sha1sums_info = [line.split() for line in f]
    
    except FileNotFoundError:
        print(f"ERROR: No {expected_sha1sums_filename} exists in {padchest_path}!")
        return

    # sort filenames using natsorted
    sha1sums_info = natsorted(sha1sums_info, key=lambda x: x[1])

    for expected_sha1, filename in sha1sums_info:
        filepath = padchest_path / filename
        
        # calculate the SHA-1 sum of the file
        sha1 = hashlib.sha1()

        print(filepath, end="", flush=True)
        try:
            with open(filepath, 'rb') as file:
                # read the file in chunks to avoid memory issues with large files
                for chunk in iter(lambda: file.read(4096), b''):
                    sha1.update(chunk)
            
            # get the calculated SHA-1 sum
            calculated_sha1 = sha1.hexdigest()
            
            # compare the calculated SHA-1 sum with the expected one
            if calculated_sha1 == expected_sha1:
                print(" ..okay")
            else:
                print(f": DIFFERENT SHA-1 SUM\n[Expected: {expected_sha1}, Calculated: {calculated_sha1}]")

        except FileNotFoundError:
            print(": File not found")

        except Exception as e:
            print(f": Error occurred - {e}")


## General utility functions
def intersect_iters(iter1: Iterable, iter2: Iterable) -> bool:
    """Determine if two iterables have any common entries

    Args:
        iter1 (Iterable): First iterable
        iter2 (Iterable): Second iterable

    Returns:
        bool: If the two iterables have common entries
    """
    intersection = set(iter1) & set(iter2)
    return bool(intersection)