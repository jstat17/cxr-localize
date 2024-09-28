import torch as th
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from pathlib import Path
import json
from collections import defaultdict
from typing import Any

from utils import evaluate

def train_one_epoch(epoch: int, num_epochs: int, model: nn.Module, train_loader: DataLoader,\
                    optimizer: optim.Optimizer, criterion: nn.Module, device: str):
    model.train()
    running_loss = 0.0
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}", unit="batch")

    for i, (images, labels) in enumerate(progress_bar):
        images, labels = images.to(device), labels.to(device)

        # zero all gradients
        optimizer.zero_grad()

        # forward pass
        outputs = model(images)
        
        # compute loss
        loss = criterion(outputs, labels.float())
        
        # backward pass and optimization
        loss.backward()
        optimizer.step()

        # update running loss
        running_loss += loss.item()
        progress_bar.set_postfix(loss=running_loss / (i + 1))

    return running_loss / len(train_loader)

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
                       device: str, num_epochs: int, save_dir: Path) -> None:
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
            device = device
        )
        print(f"Train Loss: {train_loss:.4f}")
        
        # evaluate the model on the evaluation_loader
        metrics = evaluate.evaluate_model(
            model = model,
            loader = evaluation_loader,
            split = evaluation_split,
            device = device,
            criterion = criterion
        )
        evaluate.print_metrics(metrics)

        # save the model checkpoint if evaluation loss is lower than in the previous epoch
        curr_epoch_eval_loss = metrics[f"{evaluation_split}_loss"]
        if (
            prev_epoch_eval_loss is None
                or
            curr_epoch_eval_loss < prev_epoch_eval_loss
        ):
            save_checkpoint(epoch, model, optimizer, best_checkpoint_path)
            print("-- New best --")

        save_checkpoint(epoch, model, optimizer, last_checkpoint_path)

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