import torch as th
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.models import resnet50, ResNet50_Weights
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm
import os
from pathlib import Path
import json
from collections import defaultdict
from typing import Any

from utils.dataset import PADCHEST_ABNORMALITIES_COMMON_SHENZHEN
from utils.loader import MulticlassDataset, MulticlassDatasetInMemory
from utils import dataset, evaluate

# setting parallel config
parallel = True

# Device configuration
device = "cuda"

# Number of classes (adjust this based on your dataset)
num_classes = len(PADCHEST_ABNORMALITIES_COMMON_SHENZHEN)

# Load the pre-trained ResNet-50 model with V2 weights
model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)

# Modify the last layer to output num_classes and add sigmoid for multi-label classification
model.fc = nn.Sequential(
    nn.Linear(model.fc.in_features, num_classes),
    nn.Sigmoid()
)

# Use DataParallel for multi-GPU support
if not parallel:
    device += ":0"
else:
    model = nn.DataParallel(model)

model = model.to(device)

# Loss function and optimizer
criterion = nn.BCELoss()  # For multi-label classification
optimizer = optim.AdamW(
    params = model.parameters(),
    lr = 1e-3
)

# Directories and checkpointing
home = Path.home()
padchest_path = home / "Datasets" / "PadChest"
images_path = home / "Datasets" / "PadChest-extract-common-with-Shenzhen-only-abnormality-crop" / "images-224"

save_dir = home / "models" / "ResNet50"
checkpoint_path = save_dir / "model_checkpoint.pth"
log_path = save_dir / "log.json"
os.makedirs(save_dir, exist_ok=True)

# Number of epochs and loaders
num_epochs = 100
batch_size = 64
num_workers = 8

# Data loaders
df = dataset.get_padchest_dataframe(padchest_path)
train_dataset = MulticlassDatasetInMemory(
    df = df,
    images_path = images_path,
    img_shape = 224,
    split = "train",
    split_pct1 = 0.8,
    split_pct2 = 0.9,
    possible_labels = PADCHEST_ABNORMALITIES_COMMON_SHENZHEN
)
train_loader = DataLoader(
    dataset = train_dataset,
    batch_size = batch_size,
    num_workers = num_workers,
    shuffle = True
)

test_dataset = MulticlassDatasetInMemory(
    df = df,
    images_path = images_path,
    img_shape = 224,
    split = "validate",
    split_pct1 = 0.8,
    split_pct2 = 0.9,
    possible_labels = PADCHEST_ABNORMALITIES_COMMON_SHENZHEN
)
test_loader = DataLoader(
    dataset = test_dataset,
    batch_size = 32,
    num_workers = num_workers,
    shuffle = True
)

def train_one_epoch(epoch, model, train_loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}", unit="batch")

    for i, (images, labels) in enumerate(progress_bar):
        images, labels = images.to(device), labels.to(device)

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(images)
        
        # Compute loss
        loss = criterion(outputs, labels.float())
        
        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        # Update running loss
        running_loss += loss.item()
        progress_bar.set_postfix(loss=running_loss / (i + 1))

    return running_loss / len(train_loader)

def save_checkpoint(epoch, model, optimizer, checkpoint_path):
    state = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }
    th.save(state, checkpoint_path)

def save_performance_log(performance_dict: dict[str, list[Any]], log_path: Path) -> None:
    with open(log_path, 'w') as f:
        json.dump(performance_dict, f)

def train_and_evaluate(model: nn.Module, train_loader: DataLoader, evaluation_loader: DataLoader, evaluation_split: str, optimizer: optim.Optimizer, criterion: nn.Module, device: str, num_epochs: int, save_dir: Path, log_path: Path) -> None:
    last_checkpoint_path = save_dir / "model_checkpoint.pth"
    best_checkpoint_path = save_dir / "model_best.pth"
    
    performance_curves = defaultdict(list)
    prev_epoch_eval_loss = None
    for epoch in range(num_epochs):
        # train the model for one epoch
        train_loss = train_one_epoch(epoch, model, train_loader, optimizer, criterion, device)
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
        for metric, value in metrics.items():
            performance_curves[metric].append(value)
        
        save_performance_log(performance_curves, log_path)

        # update previous evaluation loss
        if prev_epoch_eval_loss is None:
            prev_epoch_eval_loss = curr_epoch_eval_loss
        else:
            prev_epoch_eval_loss = min(curr_epoch_eval_loss, prev_epoch_eval_loss)
        

# Now train and evaluate
train_and_evaluate(
    model = model,
    train_loader = train_loader,
    evaluation_loader = test_loader,
    evaluation_split = "validate",
    optimizer = optimizer,
    criterion = criterion,
    device = device,
    num_epochs = num_epochs,
    checkpoint_path = checkpoint_path,
    log_path = log_path
)
