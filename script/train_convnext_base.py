import torch as th
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
from pathlib import Path

from dataset import padchest
from dataset.padchest import PADCHEST_ABNORMALITIES_COMMON_SHENZHEN
from utils.loader import MulticlassDataset, MulticlassDatasetInMemory
from utils.train import train_and_evaluate
from vision.convnext import get_convnext_base

# setting parallel config
parallel = True

# Device configuration
device = "cuda"

# Number of classes (adjust this based on your dataset)
num_classes = len(PADCHEST_ABNORMALITIES_COMMON_SHENZHEN)

# Load the pre-trained ResNet-50 model with ImageNet 1K V2 weights
model = get_convnext_base(
    num_classes = num_classes,
    weights = "imagenet"
)

# Use DataParallel for multi-GPU support
if not parallel:
    device += ":0"
else:
    model = nn.DataParallel(model)

model = model.to(device)

# Loss function and optimizer
criterion = nn.BCEWithLogitsLoss()  # For multi-label classification
optimizer = optim.AdamW(
    params = model.parameters(),
    lr = 1e-3
)

# Directories and checkpointing
home = Path.home()
padchest_path = home / "Datasets" / "PadChest"
images_path = home / "Datasets" / "PadChest-extract-common-with-Shenzhen-only-abnormality-crop" / "images-224"

save_dir = home / "models" / "ConvNeXt-Base"
os.makedirs(save_dir, exist_ok=True)

# Number of epochs and loaders
num_epochs = 20
batch_size = 64
num_workers = 8

# Data loaders
df = padchest.get_padchest_dataframe(padchest_path)
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

train_and_evaluate(
    model = model,
    train_loader = train_loader,
    evaluation_loader = test_loader,
    evaluation_split = "validate",
    optimizer = optimizer,
    criterion = criterion,
    device = device,
    num_epochs = num_epochs,
    save_dir = save_dir
)
