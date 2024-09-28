from argparse import ArgumentParser
from pathlib import Path

from dataset.utils import extract_files_from_gzips

"""Extract all files from gzip-compressed files.
   Run from cxr-localize:
        python -m script.extract_gzips /path/to/gzips /path/to/extracted
"""

def main(gzip_path: Path, extract_path: Path) -> None:
    extract_files_from_gzips(gzip_path, extract_path)

if __name__ == "__main__":
    # set up argument parsing
    parser = ArgumentParser(description='Extract all files from gzip-compressed files')
    parser.add_argument('gzip_path', type=str, help='Path to gzips')
    parser.add_argument('extract_path', type=str, help="Desired extract path for all files")

    # parse command line arguments
    args = parser.parse_args()
    gzip_path = Path(args.gzip_path)
    extract_path = Path(args.extract_path)

    main(gzip_path, extract_path)