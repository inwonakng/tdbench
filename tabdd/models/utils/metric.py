from typing import Literal

import torch
from torchmetrics.functional import f1_score, accuracy

"""
Metrics for prediction tasks
"""


def weighted_f1_score(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    **kwargs
):
    return f1_score(
        y_pred,
        y_true,
        num_classes=kwargs["num_classes"],
        average="weighted",
        task="multiclass",
    ).item()


def balanced_accuracy_score(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    **kwargs
):
    return accuracy(
        y_pred,
        y_true,
        num_classes=kwargs["num_classes"],
        average="weighted",
        task="multiclass",
    ).item()


PRED_METRIC_MAPPING = {
    "weighted_f1_score": weighted_f1_score,
    "balanced_accuracy_score": balanced_accuracy_score,
    "balanced_accuracy": balanced_accuracy_score,
    "f1_weighted": weighted_f1_score,
}

"""
Metrics for reconstruction tasks
"""


def recon_accuracy_score(
    target_true: torch.FloatTensor,
    target_pred: torch.FloatTensor,
    feature_mask: torch.LongTensor,
    **kwargs
):
    true_categ, pred_categ = [], []

    n_features = feature_mask.max() + 1
    for i in range(n_features):
        mask = feature_mask == i

        true_categ.append(target_true[:, mask].argmax(1))
        pred_categ.append(target_pred[:, mask].argmax(1))

    true_categ = torch.vstack(true_categ)
    pred_categ = torch.vstack(pred_categ)
    return (pred_categ == true_categ).float().mean().item()


RECON_METRIC_MAPPING = {
    "recon_accuracy_score": recon_accuracy_score,
}


def metric(
    metric_name: str, target_true: torch.Tensor, target_pred: torch.Tensor, **kwargs
):
    if target_pred.isnan().any():
        raise Exception("Metric: Predicted value has NaN values")
    if target_true.isnan().any(): 
        raise Exception("Metric: True value has NaN values")
    score = 0
    if metric_name in PRED_METRIC_MAPPING:
        score = PRED_METRIC_MAPPING[metric_name](
            y_true=target_true, y_pred=target_pred, **kwargs
        )
    elif metric_name in RECON_METRIC_MAPPING:
        score = RECON_METRIC_MAPPING[metric_name](
            target_true=target_true, target_pred=target_pred, **kwargs
        )
    else:
        raise NotImplementedError(f"Metric {metric_name} not implemented")

    return score
