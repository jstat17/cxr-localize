import torch as th
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, f1_score,\
                            roc_auc_score, average_precision_score, cohen_kappa_score, matthews_corrcoef
from tqdm import tqdm

def evaluate_model(model: nn.Module, loader: DataLoader, split: str, device: str, criterion: nn.Module) -> dict[str, float]:
    all_preds = []
    all_labels = []
    all_losses = []
    progress_bar = tqdm(
        iterable = loader,
        desc = f"Evaluating on {split} set",
        unit = "batch"
    )

    # evaluate model on all batches in loader
    model.eval()
    with th.no_grad():
        for images, labels in progress_bar:
            images, labels = images.to(device), labels.to(device)
            
            # forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            all_losses.append(loss.item())
            preds = (outputs > 0.5).float()

            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())

    # stack batches into numpy arrays, shape (num_samples, num_classes)
    all_preds = th.cat(all_preds).numpy()
    all_labels = th.cat(all_labels).numpy()

    # compute metrics
    metrics = dict()
    metrics['accuracy'] = accuracy_score(all_labels, all_preds)
    metrics['sensitivity'] = recall_score(all_labels, all_preds, average='macro')
    metrics['precision'] = precision_score(all_labels, all_preds, average='micro')
    metrics['recall'] = recall_score(all_labels, all_preds, average='micro')

    metrics['micro_f1'] = f1_score(all_labels, all_preds, average='micro')
    metrics['macro_f1 ']= f1_score(all_labels, all_preds, average='macro')
    
    metrics['auc_roc'] = roc_auc_score(all_labels, all_preds, average='macro')
    metrics['auc_pr'] = average_precision_score(all_labels, all_preds, average='macro')
    
    metrics['cohen_kappa'] = cohen_kappa_score(all_labels, all_preds)
    metrics['mcc'] = matthews_corrcoef(all_labels, all_preds)
    metrics['exact_match_ratio'] = (all_preds == all_labels).all(axis=1).mean()
    
    metrics[f'{split}_loss'] = sum(all_losses) / len(all_losses)

    return metrics

def print_metrics(metrics: dict[str: float], cols: int = 3) -> None:
    keys = list(metrics.keys())
    values = list(metrics.values())

    for i in range(0, len(keys), cols):
        row_keys = keys[i:i+cols]
        row_values = values[i:i+cols]
        row = []
        for key, value in zip(row_keys, row_values):
            row.append(f"{key:<20} {value:.4f}")
        print(" | ".join(row))