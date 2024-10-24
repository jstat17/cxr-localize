import os
import pickle as pk
from pathlib import Path
from argparse import ArgumentParser
from torch.utils.data import DataLoader

from dataset.medsam import PaddedImageDataset, get_embeddings
from vision.segment_anything.medsam import get_medsam_image_encoder

"""Save the embeddings produced by MedSAM for a dataset.
   Run from cxr-localize:
        python -m script.compute_embeddings_medsam /path/to/images /path/to/save/dir --batch_size <optional> ..args..
"""


def main(images_path: Path, save_dir: Path, batch_size: int, num_workers: int, parallel: bool, device: str) -> None:
    model = get_medsam_image_encoder()
    
    dataset = PaddedImageDataset(
        images_path = images_path,
        set_shape = (1024, 1024)
    )
    dataloader = DataLoader(
        dataset = dataset,
        batch_size = batch_size,
        shuffle = False,
        num_workers = num_workers
    )

    embeddings_dict = get_embeddings(
        model = model,
        dataloader = dataloader,
        device = device,
        parallel = parallel
    )

    save_path = save_dir / "MedSAM-Embeddings.pk"
    with open(save_path, 'wb') as f:
        pk.dump(embeddings_dict, f)

if __name__ == "__main__":
     # set up argument parsing
    parser = ArgumentParser(description='Save the embeddings produced by MedSAM for a dataset')
    parser.add_argument('images_path', type=str, help='Path to chest X-ray images')
    parser.add_argument('save_dir', type=str, help="Desired save path for the file of embeddings")
    parser.add_argument('-b', '--batch_size', type=int, default=8, help="Batch size")
    parser.add_argument('-w', '--num_workers', type=int, default=8, help="Number of workers")
    parser.add_argument('-p', '--parallel', type=str, default="True", help="Whether to evaluate in parallel on multiple GPUs")
    parser.add_argument('-d', '--device', type=str, default="cuda", help="The device to use for inference")

    # parse command line arguments
    args = parser.parse_args()
    images_path = Path(args.images_path)
    save_dir = Path(args.save_dir)

    batch_size = args.batch_size
    num_workers = args.num_workers

    parallel = args.parallel.casefold()
    match parallel:
        case "true":
            parallel = True
        case "false":
            parallel = False
        case _:
            parallel = False

    device = args.device

    main(
        images_path = images_path,
        save_dir = save_dir,
        batch_size = batch_size,
        num_workers = num_workers,
        parallel = parallel,
        device = device
    )