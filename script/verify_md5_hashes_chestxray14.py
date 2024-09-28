from argparse import ArgumentParser
from pathlib import Path

from dataset import chestxray14

"""Verify the MD5 hashes of the Chest X-ray 14 gzip files.
   Run from cxr-localize:
        python -m script.verify_md5_hashes_chestxray14 /path/to/gzips
"""

def main(chestxray14_gzip_path: Path) -> None:
    # verify md5 hashes at the specified directory
    chestxray14.verify_md5_hashes(chestxray14_gzip_path)

if __name__ == "__main__":
    # set up argument parsing
    parser = ArgumentParser(description='Verify the MD5 hashes of the Chest X-ray 14 gzip files')
    parser.add_argument('chestxray14_gzip_path', type=str, help='Path to Chest X-ray 14 gzips')

    # parse command line arguments
    args = parser.parse_args()
    chestxray14_gzip_path = Path(args.chestxray14_gzip_path)

    main(chestxray14_gzip_path)