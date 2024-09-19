import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.models import resnet50, ResNet50_Weights
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm
import os
from pathlib import Path
import json

from utils.dataset import PADCHEST_ABNORMALITIES_COMMON_SHENZHEN
from utils.loader import MulticlassDataset
from utils import dataset

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
model = nn.DataParallel(model)
model = model.to(device)

# Loss function and optimizer
criterion = nn.BCEWithLogitsLoss()  # For multi-label classification
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Directories and checkpointing
home = Path.home()
padchest_path = home / "Datasets" / "PadChest"
images_path = home / "Datasets" / "PadChest-extract-common-with-Shenzhen-only-abnormality-crop" / "images"

save_dir = home / "ResNet50"
checkpoint_path = save_dir / "model_checkpoint.pth"
log_path = save_dir / "log.json"
os.makedirs(save_dir, exist_ok=True)

# Number of epochs and loaders
num_epochs = 50

# Data loaders
df = dataset.get_padchest_dataframe(padchest_path)
train_dataset = MulticlassDataset(
    df = df,
    images_path = images_path,
    img_shape = (224, 224),
    split = "train",
    hash_percentile = 0.9,
    possible_labels = PADCHEST_ABNORMALITIES_COMMON_SHENZHEN
)
train_dataloader = DataLoader(
    dataset = train_dataset,
    batch_size = 32,
    num_workers = 4,
    shuffle = True
)

test_dataset = MulticlassDataset(
    df = df,
    images_path = images_path,
    img_shape = (224, 224),
    split = "test",
    hash_percentile = 0.9,
    possible_labels = PADCHEST_ABNORMALITIES_COMMON_SHENZHEN
)
test_dataloader = DataLoader(
    dataset = test_dataset,
    batch_size = 32,
    num_workers = 4,
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

def evaluate(model, test_loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    subset_size = 100  # Limit to a small subset for faster evaluation
    progress_bar = tqdm(test_loader, desc="Evaluating", unit="batch", total=subset_size // test_loader.batch_size)

    with torch.no_grad():
        for i, (images, labels) in enumerate(progress_bar):
            if i * test_loader.batch_size >= subset_size:
                break
            images, labels = images.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(images)
            preds = (outputs > 0.5).float()

            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())

    # Stack batches to calculate metrics
    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()

    # Compute metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='samples')
    recall = recall_score(all_labels, all_preds, average='samples')
    f1 = f1_score(all_labels, all_preds, average='samples')

    return accuracy, precision, recall, f1

def save_checkpoint(epoch, model, optimizer, checkpoint_path):
    state = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }
    torch.save(state, checkpoint_path)

def save_performance_log(performance_dict, log_path):
    with open(log_path, 'w') as f:
        json.dump(performance_dict, f)

def train_and_evaluate(model, train_loader, test_loader, optimizer, criterion, device, num_epochs, checkpoint_path, log_path):
    performance_dict = {
        'accuracy': [],
        'precision': [],
        'recall': [],
        'f1': [],
    }
    for epoch in range(num_epochs):
        # Train the model for one epoch
        train_loss = train_one_epoch(epoch, model, train_loader, optimizer, criterion, device)
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {train_loss:.4f}")
        
        # Evaluate the model on the test set
        accuracy, precision, recall, f1 = evaluate(model, test_loader, device)
        print(f"Test Metrics - Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1:.4f}")

        # Save the model checkpoint
        save_checkpoint(epoch, model, optimizer, checkpoint_path)
        print(f"Checkpoint saved at {checkpoint_path}")

        # Save performance logs
        performance_dict['accuracy'].append(accuracy)
        performance_dict['precision'].append(precision)
        performance_dict['recall'].append(recall)
        performance_dict['f1'].append(f1)
        save_performance_log(performance_dict, log_path)
        

# Now train and evaluate
train_and_evaluate(model, train_dataloader, test_dataloader, optimizer, criterion, device, num_epochs, checkpoint_path, log_path)
