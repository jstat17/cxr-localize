import torch as th
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import json
from pathlib import Path
from argparse import ArgumentParser

from dataset import padchest, chestxray14
from utils.loader import MulticlassDataset, MulticlassDatasetInMemory
from utils.train import train_and_evaluate, get_next_run_folder
from vision.resnet import get_resnet50
from vision.convnext import get_convnext_base
from vision.efficientnet import get_efficientnet_b0
from vision.swin import get_swin_s
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

def main(device: str, parallel: bool, model: nn.Module, criterion: nn.Module, optimizer: optim.Optimizer,\
         train_loader: DataLoader, evaluate_loader: DataLoader, save_dir: Path):
    # use DataParallel if parallel is enabled
    if not parallel:
        device += ":0"
    else:
        model = nn.DataParallel(model)

    model = model.to(device)
      
    train_and_evaluate(
        model = model,
        train_loader = train_loader,
        evaluation_loader = evaluate_loader,
        evaluation_split = "validate",
        optimizer = optimizer,
        criterion = criterion,
        device = device,
        num_epochs = num_epochs,
        save_dir = save_dir
    )

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
    parser.add_argument('-n', '--num_epochs', type=int, default=100, help="Number of epochs to train for")
    parser.add_argument('-b', '--batch_size', type=int, default=64, help="Batch size")
    parser.add_argument('-lr', '--learning_rate', type=str, default=1e-3, help="Initial learning rate")
    parser.add_argument('-l', '--loss', type=str, default='bce', help="Loss function")
    parser.add_argument('-p1', '--split_pct1', type=float, default=0.8, help="The training split percent")
    parser.add_argument('-p2', '--split_pct2', type=float, default=0.9, help="The training and validation split percent")
    parser.add_argument('-w1', '--workers_train', type=int, default=32, help="Number of training dataloader workers")
    parser.add_argument('-w2', '--workers_validate', type=int, default=8, help="Number of validation dataloader workers")
    parser.add_argument('-wg', '--weights', type=str, default='imagenet', help="Initial weights to use for the model ('imagenet' or 'None')")
    parser.add_argument('-m', '--load_memory', type=bool, default=False, help="Whether to load all images into memory")
    parser.add_argument('-i', '--info', type=str, default="", help="Information for logging")

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

    num_epochs = args.num_epochs
    batch_size = args.batch_size
    learning_rate = float(args.learning_rate)
    loss = args.loss.casefold()

    split_pct1 = args.split_pct1
    split_pct2 = args.split_pct2

    workers_train = args.workers_train
    workers_validate = args.workers_validate

    load_memory = args.load_memory

    weights = args.weights.casefold()
    if weights == "none":
        weights = None

    info = args.info

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
                case "base":
                    model = get_convnext_base(num_classes, weights)

        case "efficientnet":
            match model_size:
                case "b0":
                    model = get_efficientnet_b0(num_classes, weights)

        case "swin":
            match model_size:
                case "s":
                    model = get_swin_s(num_classes, weights)

    # set loss function
    match loss:
        case 'bce':
            criterion = nn.BCEWithLogitsLoss()
        case 'wbce':
            criterion = WeightedBCEWithLogitsLoss()

    # set optimizer
    optimizer = optim.AdamW(
        params = model.parameters(),
        lr = learning_rate
    )

    # data loaders
    if load_memory:
        dataset_class = MulticlassDatasetInMemory
    else:
        dataset_class = MulticlassDataset

    train_dataset = dataset_class(
        df = df,
        images_path = images_path,
        img_shape = image_size,
        split = "train",
        split_pct1 = split_pct1,
        split_pct2 = split_pct2,
        possible_labels = possible_labels
    )
    train_loader = DataLoader(
        dataset = train_dataset,
        batch_size = batch_size,
        num_workers = workers_train,
        shuffle = True
    )

    validate_dataset = MulticlassDatasetInMemory(
        df = df,
        images_path = images_path,
        img_shape = image_size,
        split = "validate",
        split_pct1 = split_pct1,
        split_pct2 = split_pct2,
        possible_labels = possible_labels
    )
    validate_loader = DataLoader(
        dataset = validate_dataset,
        batch_size = 16,
        num_workers = workers_validate,
        shuffle = True
    )

    # creating save directory
    save_dir_root = MODELS_PATH / model.fullname
    os.makedirs(save_dir_root, exist_ok=True)

    save_dir = get_next_run_folder(save_dir_root)

    # save hyperparams log
    hyperparams = {
        'model_name': model_name,
        'model_size': model_size,
        'dataset': dataset,
        'image_size': image_size,
        'device': device,
        'parallel': parallel,
        'num_epochs': num_epochs,
        'batch_size': batch_size,
        'learning_rate': learning_rate,
        'split_pct1': split_pct1,
        'split_pct2': split_pct2,
        'workers_train': workers_train,
        'workers_validate': workers_validate,
        'weights': weights,
        'info': info,
        'criterion': str(criterion),
        'optimizer': str(optimizer)
    }
    
    hyperparams_log_path = save_dir / "hyperparams.json"
    with open(hyperparams_log_path, 'w') as f:
        json.dump(hyperparams, f, indent=2)
    
    main(
        device = device,
        parallel = parallel,
        model = model,
        criterion = criterion,
        optimizer = optimizer,
        train_loader = train_loader,
        evaluate_loader = validate_loader,
        save_dir = save_dir
    )