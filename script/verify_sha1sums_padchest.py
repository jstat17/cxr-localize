from utils import dataset
from argparse import ArgumentParser
from pathlib import Path

def main():
    # set up argument parsing
    parser = ArgumentParser(description='Verify SHA-1 sums of the originall-downloaded PadChest files.')
    parser.add_argument('padchest_path', type=str, help='Path to originally-downloaded PadChest dataset')

    # parse command line arguments
    args = parser.parse_args()
    padchest_path = Path(args.padchest_path)

    # verify sha-1 sums at the specified PadChest directory
    dataset.verify_sha1sums(padchest_path)

if __name__ == "__main__":
    main()