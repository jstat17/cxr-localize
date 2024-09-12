import os
from pathlib import Path
from argparse import ArgumentParser
import numpy as np
import torch as th
from torch.utils.data import DataLoader
from torch.nn import DataParallel
import skimage
import torchxrayvision as xrv
from concurrent.futures import ThreadPoolExecutor, wait, ALL_COMPLETED

from utils.loader import XRVDataset
from utils import transform

"""Segment and crop out the lungs from images using TorchXRayVision.
   Run from cxr-localize:
        python -m script.crop_lungs /path/to/images /path/to/cropped --batch_size <optional> -- num_workers <optional> --parallel <optional> --force <optional>
"""

def save_crop_and_mask(images_path: Path, save_path_images: Path, save_path_masks: Path, save_path_logs: Path,\
                       x1: int, y1: int, x2: int, y2: int, filename: str, shape_current: tuple[int, int],\
                       shape_original: tuple[int, int], lungs: np.ndarray) -> None:    
    # resize bounding box to the original image's size
    (x1, y1, x2, y2) = transform.scale_bounding_box(
        x1 = x1,
        y1 = y1, 
        x2 = x2,
        y2 = y2,
        shape_old = shape_current,
        shape_new = shape_original
    )
    
    # make the bounding box square and fit to within the original image
    (x1, y1, x2, y2) = transform.make_bounding_box_square(
        x1 = x1,
        y1 = y1, 
        x2 = x2,
        y2 = y2,
        shape = shape_original,
        allowance = 0.05
    )
    
    # crop original image
    original_image_path = images_path / filename
    original_image = skimage.io.imread(original_image_path)
    original_image_cropped = original_image[x1:x2, y1:y2]
    
    # save image
    try:
        skimage.io.imsave(save_path_images / filename, original_image_cropped)

    except IndexError as e:
        print(f"IndexError in saving cropped image: {e}")

    except OSError:
        print(f"OSError in saving cropped image, crop could be blank: {e}")

    # save log
    with open(save_path_logs / f"{filename.split('.')[0]}.txt", 'w') as f:
        f.write(str({'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2}))

    # save lungs mask after resizing them
    try:
        lungs = transform.resize(lungs, shape_original).astype(bool)
        skimage.io.imsave(save_path_masks / filename, lungs)

    except IndexError as e:
        print(f"IndexError in saving lungs: {e}")
        
    except OSError:
        print(f"OSError in saving lungs, segmentation map could be blank: {e}")

def main(images_path: Path, save_path: Path, batch_size: int = 32, num_workers: int = 1,\
         device: str = 'cuda', parallel: bool = True, force: bool = False):
    th.cuda.empty_cache()
    
    # prepare loading and saving paths
    save_path_masks = save_path / "masks"
    save_path_images = save_path / "images"
    save_path_logs = save_path / "logs"
    os.makedirs(save_path_masks, exist_ok=True)
    os.makedirs(save_path_images, exist_ok=True)
    os.makedirs(save_path_logs, exist_ok=True)

    # prepare dataset loader
    if force:
        dataset = XRVDataset(images_path)
    else:
        dataset = XRVDataset(images_path, save_path_images)

    dataloader = DataLoader(
        dataset = dataset,
        batch_size = batch_size,
        num_workers = num_workers,
        shuffle = False
    )

    segm_model = xrv.baseline_models.chestx_det.PSPNet()
    right_lung_idx = segm_model.targets.index("Right Lung")
    left_lung_idx = segm_model.targets.index("Left Lung")

    if parallel:
        segm_model = DataParallel(segm_model)
    else:
        device += ":0"

    segm_model = segm_model.to(device)

    for batch in dataloader:
        tensor_batch, original_sizes, filenames = batch
        tensor_batch = tensor_batch.to(device)
        
        with th.no_grad():
            pred_batch = segm_model(tensor_batch)
        
            # apply sigmoid to predictions
            pred_batch = 1 / (1 + th.exp(-pred_batch))
            pred_batch[pred_batch < 0.5] = 0
            pred_batch[pred_batch > 0.5] = 1

            # create full lung mask from right and left lobes
            right_lung_batch = pred_batch[:, right_lung_idx]
            left_lung_batch = pred_batch[:, left_lung_idx]
            lungs_batch = right_lung_batch + left_lung_batch

            # create bounding boxes from all lungs in batch
            try:
                bounding_boxes_batch = transform.get_bounding_box_from_mask(lungs_batch)
            except RuntimeError as e: # some bug in pytorch which can cause this when creating the non-zero indices
                print(f"RuntimeError in creating bounding boxes from batch of masks: {e}")
            
            # move to cpu and numpy
            bounding_boxes_batch = bounding_boxes_batch.cpu().numpy()
            lungs_batch = lungs_batch.cpu().to(th.uint8).numpy()

            # iterate through each file and pass to ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=batch_size) as executor:
                futures = []
                for i, filename in enumerate(filenames):
                    # extract bounding box and lungs for current image
                    (x1, y1, x2, y2) = bounding_boxes_batch[i].tolist()
                    height, width = original_sizes[i].tolist()
                    shape_original = (height, width)
                    lungs = lungs_batch[i]

                    kwargs = {
                        'images_path': images_path,
                        'save_path_images': save_path_images,
                        'save_path_masks': save_path_masks,
                        'save_path_logs': save_path_logs,
                        'x1': x1,
                        'y1': y1,
                        'x2': x2,
                        'y2': y2,
                        'filename': filename,
                        'shape_current': XRVDataset.set_shape,
                        'shape_original': shape_original,
                        'lungs': lungs
                    }
                    futures.append(
                        executor.submit(save_crop_and_mask, **kwargs)
                    )

                try:
                    wait(futures, return_when=ALL_COMPLETED)
                except Exception as e:
                    print(f"An error occurred within the ThreadPoolExecutor: {e}")

    th.cuda.empty_cache()

if __name__ == "__main__":
    # set up argument parsing
    parser = ArgumentParser(description='Segment and crop out the lungs from images using TorchXRayVision')
    parser.add_argument('images_path', type=str, help='Path to chest X-ray images')
    parser.add_argument('save_path', type=str, help="Desired save path for crops and masks")
    parser.add_argument('--batch_size', type=int, default=32, help="Batch size of images to load at once")
    parser.add_argument('--num_workers', type=int, default=1, help="Number of CPU workers to use for the data loader")
    parser.add_argument('--parallel', type=bool, default=True, help="Split computation across multiple GPUs in parallel")
    parser.add_argument('--force', type=bool, default=False, help="Force cropping even if images have already been cropped previously")

    # parse command line arguments
    args = parser.parse_args()
    images_path = Path(args.images_path)
    save_path = Path(args.save_path)
    batch_size = args.batch_size
    num_workers = args.num_workers
    parallel = args.parallel
    force = args.force

    main(
        images_path = images_path,
        save_path = save_path,
        batch_size = batch_size,
        num_workers = num_workers,
        parallel = parallel,
        force = force
    )