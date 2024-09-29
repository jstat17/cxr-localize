import torch as th
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, f1_score,\
                            roc_auc_score, average_precision_score, cohen_kappa_score, matthews_corrcoef,\
                            jaccard_score, hamming_loss, confusion_matrix
from tqdm import tqdm
import numpy as np
from collections.abc import Callable
from typing import Any

def evaluate_model(model: nn.Module, loader: DataLoader, split: str, device: str, criterion: nn.Module) -> dict[str, float]:
    all_probas = []
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
            probas = th.sigmoid(outputs)
            preds = (probas > 0.5).float()

            all_probas.append(probas.cpu())
            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())

    # stack batches into numpy arrays, shape (num_samples, num_classes)
    all_probas = th.cat(all_probas).numpy().astype(float)
    all_preds = th.cat(all_preds).numpy().astype(int)
    all_labels = th.cat(all_labels).numpy().astype(int)

    # compute metrics
    metrics = dict()

    # accuracy
    metrics['accuracy_sample'] = accuracy_score(all_labels, all_preds) # all labels predicted correctly for an image
    metrics['accuracy_label'] = get_metric_macro(all_labels, all_preds, accuracy_score) # number of labels predicted correctly

    # sensitivity
    metrics['sensitivity_macro'] = recall_score(all_labels, all_preds, average='macro')
    metrics['sensitivity_micro'] = recall_score(all_labels, all_preds, average='micro')

    # precision
    metrics['precision_macro'] = precision_score(all_labels, all_preds, average='macro')
    metrics['precision_micro'] = precision_score(all_labels, all_preds, average='micro')

    # specificity
    metrics['specificity_macro'] = get_metric_macro(all_labels, all_preds, specificity_score)

    # F1
    metrics['f1_macro']= f1_score(all_labels, all_preds, average='macro')
    metrics['f1_micro'] = f1_score(all_labels, all_preds, average='micro')
    
    # metrics that require scores instead of classifications
    metrics['auc_roc_macro'] = get_metric_macro(all_labels, all_probas, roc_auc_score)
    metrics['auc_pr_macro'] = get_metric_macro(all_labels, all_probas, average_precision_score)
    
    metrics['cohen_kappa'] = get_metric_macro(all_labels, all_preds, cohen_kappa_score)
    metrics['mcc'] = get_metric_macro(all_labels, all_preds, matthews_corrcoef)
    
    metrics['jaccard_similarity'] = get_metric_macro(all_labels, all_preds, jaccard_score, average='binary')
    metrics['hamming_loss'] = get_metric_macro(all_labels, all_preds, hamming_loss)

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

def get_metric_macro(y_true, y_pred, metric: Callable, **kwargs: Any) -> float:
    # solves problem where there could be only one class in the true labels
    metrics = []
    for i in range(y_true.shape[1]):
        if len(set(y_true[:, i])) > 1:  # check if there is more than one class
            metric = metric(y_true[:, i], y_pred[:, i], **kwargs)
            metrics.append(metric)

    return np.mean(metrics)

def specificity_score(y_true, y_pred):
    """
    Compute specificity from true and predicted labels.

    Parameters:
    y_true (list or array-like): True binary labels.
    y_pred (list or array-like): Predicted binary labels.

    Returns:
    float: Specificity value.
    """
    # Calculate confusion matrix
    tn, fp, _fn, _tp = confusion_matrix(y_true, y_pred).ravel()
    
    # Calculate specificity
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0  # Avoid division by zero
    return specificity