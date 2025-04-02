from typing import Tuple, Optional
import numpy as np

import torch.nn.functional as F
import torch
from .utils import get_mlp


def get_forget_counts(
    X: np.ndarray,
    y: np.ndarray,
    n_epochs: int,
    mlp_dim: int,
    n_hidden_layers: int,
    lr: float,
    random_state: int = 0,
) -> np.ndarray:
    n_labels = len(np.unique(y))
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"

    clf = get_mlp(
        input_shape=X.shape[1],
        n_hidden_layers=n_hidden_layers,
        hidden_dim=mlp_dim,
        n_labels=n_labels,
        random_state=random_state,
    ).to(device)

    # use torch versions of the data
    X_pt = torch.tensor(X, dtype=torch.float32).to(device)
    y_pt = torch.tensor(y, dtype=torch.long).to(device)
    forget_counts = torch.zeros(y_pt.size(0)).to(device)
    optimizer = torch.optim.Adam(clf.parameters(), lr=lr)

    # training loop
    early_stopping_patience = 10
    bad_update_count = 0
    best_loss = np.inf
    for epoch in range(n_epochs):
        optimizer.zero_grad()
        logits = clf(X_pt)
        loss = F.cross_entropy(logits, y_pt)
        loss.backward()
        optimizer.step()
        forget_counts[logits.argmax(1) != y_pt] += 1
        # if loss is not decreasing, stop
        if epoch > 0 and loss > best_loss:
            bad_update_count += 1
            if bad_update_count >= early_stopping_patience:
                break
        else:
            bad_update_count = 0
            best_loss = loss

    # get the indices of the samples that were forgotten the most
    forget_counts = forget_counts.cpu().numpy()
    return forget_counts


def forgetting(
    X: np.ndarray,
    y: np.ndarray,
    N: int,
    n_epochs: int,
    mlp_dim: int,
    n_hidden_layers: int,
    lr: float,
    match_balance: bool = False,
    random_state: int | np.random.Generator = 0,
) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    forget_counts = get_forget_counts(
        X=X,
        y=y,
        n_epochs=n_epochs,
        mlp_dim=mlp_dim,
        n_hidden_layers=n_hidden_layers,
        lr=lr,
        random_state=random_state,
    )
    sampled_X, sampled_y = [], []
    y_unique, y_counts = np.unique(y, return_counts=True)

    if N > 1:
        if match_balance:
            sample_sizes = (y_counts / y_counts.sum() * N).astype(int)
        else:
            sample_sizes = [int(N)] * len(y_unique)
    else:
        if match_balance:
            sample_sizes = (y_counts * N).astype(int)
        else:
            sample_sizes = [int(len(y) * N)] * len(y_unique)

    for label, ss in zip(y_unique, sample_sizes):
        sampled_X.append(X[y == label][forget_counts[y == label].argsort()[::-1][:ss]])
        sampled_y += [label] * ss

    sampled_X = np.vstack(sampled_X)
    sampled_y = np.array(sampled_y)

    return sampled_X, sampled_y
