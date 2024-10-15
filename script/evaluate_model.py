import torch as th
import torch.nn as nn
from torch.utils.data import DataLoader
import os
import json
from pathlib import Path
from argparse import ArgumentParser

from dataset import padchest, chestxray14
from dataset.utils import read_file_lines
from utils.evaluate import evaluate_model
from utils.loader import MulticlassDataset
from vision.resnet import get_resnet50
from vision.convnext import get_convnext_b, get_convnext_s
from vision.efficientnet import get_efficientnet_b0, get_efficientnet_b4
from vision.swin import get_swin_s
from vision.vit import get_vit_s
from loss.wbce import WeightedBCEWithLogitsLoss

"""Train a vision model on the given dataset/s.
   Run from cxr-localize:
        python -m script.train ..args..
"""

# directory constants
HOME = Path.home()
DATASETS_PATH = HOME / "Datasets"
PADCHEST_CONF_PATH = DATASETS_PATH / "PadChest"
CXR14_CONF_PATH = DATASETS_PATH / "ChestX-ray14"
MODELS_PATH = HOME / "models"
EVALUATION_PATH = MODELS_PATH / "evaluate"

def main(device: str, parallel: bool, model: nn.Module, criterion: nn.Module, evaluate_loader: DataLoader,\
         save_dir: Path):
    # use DataParallel if parallel is enabled
    if not parallel:
        device += ":0"
    else:
        model = nn.DataParallel(model)

    model = model.to(device)
      
    metrics, raw = evaluate_model(
        model = model,
        loader = evaluate_loader,
        split = "evaluate",
        device = device,
        criterion = criterion
    )

    # save evaluation outputs to json
    output = {
        'metrics': metrics,
        'raw': raw
    }
    output_save_path = save_dir / f"{model.fullname}-evaluation.json"
    with open(output_save_path, 'w') as f:
        json.dump(output, f, indent=2)

if __name__ == "__main__":
    # set up argument parsing
    parser = ArgumentParser(description='Train a vision model on the given dataset/s.')
    parser.add_argument('model_name', type=str, help='Model to train')
    parser.add_argument('model_size', type=str, help='Model size')
    parser.add_argument('-img', '--images_path', nargs='+', type=str, default=[str(HOME)], help='Path to images')
    parser.add_argument('-ds', '--dataset', nargs='+', type=str, default=['padchest'], help="Training dataset/s ('padchest', 'chestxray14')")
    parser.add_argument('-s', '--size', type=int, default=224, help="Image size")
    parser.add_argument('-d', '--device', type=str, default='cuda', help="The device used to train the model ('cpu' or 'cuda')")
    parser.add_argument('-p', '--parallel', type=bool, default=True, help="Whether to train in parallel on multiple GPUs")
    parser.add_argument('-b', '--batch_size', type=int, default=64, help="Batch size")
    parser.add_argument('-l', '--loss', type=str, default='bce', help="Loss function")
    parser.add_argument('-p1', '--split_pct1', type=float, default=0.8, help="The training split percent")
    parser.add_argument('-p2', '--split_pct2', type=float, default=0.9, help="The training and validation split percent")
    parser.add_argument('-of', '--official_split', type=bool, default=False, help="Whether to use the official training split (ignores -p2, -p2)")
    parser.add_argument('-wv', '--workers_evaluate', type=int, default=8, help="Number of validation dataloader workers")
    parser.add_argument('-wg', '--weights', type=str, default='imagenet', help="Weights to use for the model ('imagenet', 'None' or path/to/weights)")

    # parse command line arguments
    args = parser.parse_args()

    model_name = args.model_name.casefold()
    model_size = args.model_size.casefold()

    # handle if ~ is entered
    images_path = args.images_path[0]
    if images_path[0:2] == "~/":
        images_path = HOME / args.images_path
    else:
        images_path = Path(images_path)

    dataset = args.dataset[0].casefold()
    image_size = args.size

    device = args.device
    parallel = args.parallel

    batch_size = args.batch_size
    loss = args.loss.casefold()

    split_pct1 = args.split_pct1
    split_pct2 = args.split_pct2
    official_split = args.official_split

    workers_evaluate = args.workers_evaluate

    weights = args.weights
    if "/" in weights:
        state_dict_path = Path(weights)
        weights = th.load(state_dict_path)
    
    else:
        weights = weights.casefold()
        if weights == "none":
            weights = None

    # get dataset info
    match dataset:
        case "padchest":
            df = padchest.get_padchest_dataframe(PADCHEST_CONF_PATH)
            possible_labels = padchest.PADCHEST_ABNORMALITIES_COMMON_SHENZHEN
        
        case "chestxray14":
            df = chestxray14.get_chestxray14_dataframe(CXR14_CONF_PATH)
            possible_labels = chestxray14.CXR_14_ABNORMALITIES

    num_classes = len(possible_labels)

    # load model based on arguments
    match model_name:
        case "resnet":
            match model_size:
                case "50":
                    model = get_resnet50(num_classes, weights)

        case "convnext":
            match model_size:
                case "s":
                    model = get_convnext_s(num_classes, weights)
                case "b":
                    model = get_convnext_b(num_classes, weights)

        case "efficientnet":
            match model_size:
                case "b0":
                    model = get_efficientnet_b0(num_classes, weights)
                case "b4":
                    model = get_efficientnet_b4(num_classes, weights)

        case "swin":
            match model_size:
                case "s":
                    model = get_swin_s(num_classes, weights)

        case "vit":
            match model_size:
                case "s":
                    model = get_vit_s(num_classes, weights)

    # set loss function
    match loss:
        case 'bce':
            criterion = nn.BCEWithLogitsLoss()
        case 'wbce':
            criterion = WeightedBCEWithLogitsLoss()

    # consider using official split of the dataset
    if official_split:
        match dataset:
            case "chestxray14":
                available_image_filenames = set(os.listdir(images_path))

                train_filenames_path = CXR14_CONF_PATH / 'train_val_list.txt'
                train_filenames = read_file_lines(train_filenames_path)
                train_filenames = [filename for filename in train_filenames if filename not in chestxray14.IGNORED_IMAGES]
                train_filenames = [filename for filename in train_filenames if filename in available_image_filenames]

                validate_filenames_path = CXR14_CONF_PATH / 'test_list.txt'
                validate_filenames = read_file_lines(validate_filenames_path)
                validate_filenames = [filename for filename in validate_filenames if filename not in chestxray14.IGNORED_IMAGES]
                validate_filenames = [filename for filename in validate_filenames if filename in available_image_filenames]

            case "padchest":
                train_filenames = None
                validate_filenames = None

    else:
        train_filenames = None
        validate_filenames = None

    evaluate_dataset = MulticlassDataset(
        df = df,
        images_path = images_path,
        img_shape = image_size,
        split = "validate",
        split_pct1 = split_pct1,
        split_pct2 = split_pct2,
        possible_labels = possible_labels,
        filenames = validate_filenames
    )
    evaluate_loader = DataLoader(
        dataset = evaluate_dataset,
        batch_size = batch_size,
        num_workers = workers_evaluate,
        shuffle = True
    )

    # creating save directory
    os.makedirs(EVALUATION_PATH, exist_ok=True)
    
    main(
        device = device,
        parallel = parallel,
        model = model,
        criterion = criterion,
        evaluate_loader = evaluate_loader,
        save_dir = EVALUATION_PATH
    )