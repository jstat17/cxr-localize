from argparse import ArgumentParser
from pathlib import Path

from dataset import padchest

"""Verify the SHA-1 sums of the PadChest dataset files.
   Run from cxr-localize:
        python -m script.verify_sha1sums_padchest /path/to/PadChest
"""

def main(padchest_path: Path) -> None:
    # verify sha-1 sums at the specified PadChest directory
    padchest.verify_sha1sums(padchest_path)

if __name__ == "__main__":
    # set up argument parsing
    parser = ArgumentParser(description='Verify SHA-1 sums of the originall-downloaded PadChest files')
    parser.add_argument('padchest_path', type=str, help='Path to originally-downloaded PadChest dataset')

    # parse command line arguments
    args = parser.parse_args()
    padchest_path = Path(args.padchest_path)

    main(padchest_path)