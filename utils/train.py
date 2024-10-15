import torch as th
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from pathlib import Path
import natsort
import json
from collections import defaultdict
from typing import Any

from utils import evaluate

def train_one_epoch(epoch: int, num_epochs: int, model: nn.Module, train_loader: DataLoader,\
                    optimizer: optim.Optimizer, criterion: nn.Module, device: str, accumulation_factor: int = 1):
    """Train the model for one epoch.

    Args:
        epoch (int): The current epoch number.
        num_epochs (int): The total number of epochs.
        model (nn.Module): The model to be trained.
        train_loader (DataLoader): The data loader for the training set.
        optimizer (optim.Optimizer): The optimizer to be used.
        criterion (nn.Module): The loss function to be used.
        device (str): The device to be used ('cuda' or 'cpu').
        accumulation_factor (int, optional): The gradient accumulation factor. Defaults to 1.

    Returns:
        _type_: _description_
    """
    model.train()
    running_loss = 0.0
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}", unit="batch")

    # initialize the gradient accumulation counter
    accumulation_counter = 0
    n_batches = len(train_loader)

    for i, (images, labels) in enumerate(progress_bar):
        images, labels = images.to(device), labels.to(device)

        # zero all gradients
        if accumulation_counter == 0:
            optimizer.zero_grad()

        # forward pass
        outputs = model(images)
        
        # compute loss
        loss = criterion(outputs, labels.float())
        
        # backward pass
        loss.backward()

        # Increment the accumulation counter
        accumulation_counter += 1

        # If the accumulation factor is reached or it's the last batch, perform optimization
        if accumulation_counter == accumulation_factor or i == n_batches - 1:
            optimizer.step()
            accumulation_counter = 0

        # update running loss
        running_loss += loss.item()
        progress_bar.set_postfix(loss=running_loss / (i + 1))

    return running_loss / n_batches


def save_checkpoint(epoch: int, model: nn.Module, optimizer: optim.Optimizer, checkpoint_path: Path):
    state = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }
    th.save(state, checkpoint_path)

def save_performance_log(performance_dict: dict[str, list[Any]], log_path: Path) -> None:
    with open(log_path, 'w') as f:
        json.dump(performance_dict, f, indent=2)

def train_and_evaluate(model: nn.Module, train_loader: DataLoader, evaluation_loader: DataLoader,\
                       evaluation_split: str, optimizer: optim.Optimizer, criterion: nn.Module,\
                       device: str, num_epochs: int, accumulation_factor: int, save_dir: Path) -> None:
    last_checkpoint_path = save_dir / "model_checkpoint.pth"
    best_checkpoint_path = save_dir / "model_best.pth"
    log_path = save_dir / "log.json"

    performance_curves = defaultdict(list)
    prev_epoch_eval_loss = None
    for epoch in range(num_epochs):
        # train the model for one epoch
        train_loss = train_one_epoch(
            epoch = epoch,
            num_epochs = num_epochs,
            model = model,
            train_loader = train_loader,
            optimizer = optimizer,
            criterion = criterion,
            device = device,
            accumulation_factor = accumulation_factor
        )
        print(f"Train Loss: {train_loss:.4f}")
        
        # evaluate the model on the evaluation_loader
        metrics, _raw = evaluate.evaluate_model(
            model = model,
            loader = evaluation_loader,
            split = evaluation_split,
            device = device,
            criterion = criterion
        )
        evaluate.print_metrics(metrics)

        # save the last model checkpoint
        save_checkpoint(epoch, model, optimizer, last_checkpoint_path)

        # save the best model checkpoint if evaluation loss is lower than in the previous epoch
        curr_epoch_eval_loss = metrics[f"{evaluation_split}_loss"]
        if (
            prev_epoch_eval_loss is None
                or
            curr_epoch_eval_loss < prev_epoch_eval_loss
        ):
            save_checkpoint(epoch, model, optimizer, best_checkpoint_path)
            print("-- New best --")

        # if the evaluation loss is worse, decrease learning rate by an order of magnitude
        # else:
        #     optimizer.param_groups[0]['lr'] /= 10

        # save performance logs
        performance_curves['epoch'].append(epoch)
        performance_curves['train_loss'].append(train_loss)
        for metric, value in metrics.items():
            performance_curves[metric].append(value)
        
        save_performance_log(performance_curves, log_path)

        # update previous evaluation loss
        if prev_epoch_eval_loss is None:
            prev_epoch_eval_loss = curr_epoch_eval_loss
        else:
            prev_epoch_eval_loss = min(curr_epoch_eval_loss, prev_epoch_eval_loss)

def get_next_run_folder(runs_dir: Path) -> Path:
    """Create a new run folder and return the path to it

    Args:
        runs_dir (Path): The directory of training runs

    Returns:
        Path: The path of the new run folder
    """
    # get a list of all folders in the root directory
    folders = [f for f in runs_dir.iterdir() if f.is_dir()]

    # filter the list to only include folders that match the run_XXX pattern
    run_folders = [f for f in folders if f.name.startswith('run_') and f.name[4:].isdigit()]

    # sort the list of run folders using natsort
    run_folders = natsort.natsorted(run_folders, key=lambda x: int(x.name[4:]))

    # if there are no run folders, create the first one
    if not run_folders:
        next_run_folder = runs_dir / 'run_001'

    else:
        # get the last run folder and extract its number
        last_run_folder = run_folders[-1]
        last_run_number = int(last_run_folder.name[4:])

        # create the next run folder
        next_run_number = last_run_number + 1
        next_run_folder = runs_dir / f'run_{next_run_number:03d}'

    # create the next run folder if it doesn't exist
    next_run_folder.mkdir(exist_ok=True)

    return next_run_folder
