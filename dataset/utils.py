import gzip
import os
import tarfile
from pathlib import Path
from collections.abc import Iterable
from typing import Any

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

def get_iter_to_iter_dict(iter1: Iterable[Any], iter2: Iterable[Any]) -> dict[Any, Any]:
    """Get a dictionary from iter1 values to iter2 values

    Args:
        iter1 (Iterable[Any]): First iterable (dictionary keys)
        iter2 (Iterable[Any]): Second iterable (dictionary values)

    Returns:
        dict[Any, Any]: Dictionary from iter1 values to iter2 values
    """
    mapping = dict()
    for iter1_val, iter2_val in zip(iter1, iter2):
        mapping[iter1_val] = iter2_val
    
    return mapping

def extract_files_from_gzips(gzip_path: Path, extract_path: Path) -> None:
    """Extract all files from gzip-compressed files
    
    Args:
        gzip_path (Path): Path to gzips
        extract_path (Path): Desired extract path for all files
    """
    os.makedirs(extract_path, exist_ok=True)
    existing_filenames = set(
        os.listdir(extract_path)
    )
    gzip_filenames = os.listdir(gzip_path)

    # iterate through all gzips
    for filename in gzip_filenames:
        gzip_path = gzip_path / filename
        n_extracted = 0

        # read and decompress the gzip file into memory
        with gzip.open(gzip_path, 'rb') as gz:
            # open the tar file from the decompressed gzip content
            with tarfile.open(fileobj=gz, mode='r:') as tar:
                for tar_member in tar.getmembers():
                    # get the base name of the file, which removes folder structure
                    tar_member_filename = Path(tar_member.name).name

                    # extract tar member if it has not been extracted before
                    if tar_member_filename not in existing_filenames:
                        file_obj = tar.extractfile(tar_member)

                        if file_obj is not None:
                            image_path = extract_path / tar_member_filename
                            with open(image_path, 'wb') as out_file:
                                out_file.write(file_obj.read())

                            n_extracted += 1

        print(f"Extracted {n_extracted} file/s from {gzip_path}")