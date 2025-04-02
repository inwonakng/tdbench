from typing import Tuple, Optional
import numpy as np

import torch.nn.functional as F
import torch
from .utils import get_mlp


def get_grand_score(
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
    optimizer = torch.optim.Adam(clf.parameters(), lr=lr)

    # training loop
    early_stopping_patience = 10
    bad_update_count = 0
    best_loss = np.inf
    for epoch in range(n_epochs):
        optimizer.zero_grad()
        embeddings = clf.embedder(X_pt)
        logits = clf.classifier(embeddings)
        loss = F.cross_entropy(logits, y_pt)
        loss.backward()
        optimizer.step()
        # if loss is not decreasing, stop
        if epoch > 0 and loss > best_loss:
            bad_update_count += 1
            if bad_update_count >= early_stopping_patience:
                break
        else:
            bad_update_count = 0
            best_loss = loss

    # now we are done training. Need to collect the grand scores here.
    clf.eval()
    optimizer.zero_grad()
    logits = clf(X_pt)
    loss = F.cross_entropy(logits.requires_grad_(True), y_pt)
    with torch.no_grad():
        param_grads = torch.autograd.grad(loss, logits)[0]
        grand_scores = torch.norm(
            torch.cat(
                [
                    param_grads,
                    (
                        (
                            embeddings.view(X_pt.size(0), 1, embeddings.size(1)).repeat(
                                1, n_labels, 1
                            )
                        )
                        * (
                            param_grads.view(X_pt.size(0), n_labels, 1).repeat(
                                1, 1, embeddings.size(1)
                            )
                        )
                    ).view(X_pt.size(0), -1),
                ],
                dim=1,
            ),
            dim=1,
            p=2,
        )

    return grand_scores.detach().cpu().numpy()


def grand(
    X: np.ndarray,
    y: np.ndarray,
    N: int,
    n_repeat: int,
    n_epochs: int,
    mlp_dim: int,
    n_hidden_layers: int,
    lr: float,
    match_balance: bool = False,
    random_state: int | np.random.Generator = 0,
) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    grand_scores = []
    for _ in range(n_repeat):
        one_grand_scores = get_grand_score(
            X=X,
            y=y,
            n_epochs=n_epochs,
            mlp_dim=mlp_dim,
            n_hidden_layers=n_hidden_layers,
            lr=lr,
            random_state=random_state,
        )
        grand_scores.append(one_grand_scores[:,None])
    grand_scores = np.concatenate(grand_scores, axis=1).mean(axis=1)

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
        sampled_X.append(X[y == label][grand_scores[y == label].argsort()[::-1][:ss]])
        sampled_y += [label] * ss

    sampled_X = np.vstack(sampled_X)
    sampled_y = np.array(sampled_y)

    return sampled_X, sampled_y
