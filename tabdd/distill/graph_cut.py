from typing import Tuple, Optional
import numpy as np

import torch.nn.functional as F
import torch
from .utils import get_mlp


def cossim_np(v1, v2):
    num = np.dot(v1, v2.T)
    denom = np.linalg.norm(v1, axis=1).reshape(-1, 1) * np.linalg.norm(v2, axis=1)
    res = num / denom
    res[np.isneginf(res)] = 0.0
    return 0.5 + 0.5 * res


def get_device():
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    return device


def train_mlp(
    X: np.ndarray,
    y: np.ndarray,
    n_epochs: int,
    mlp_dim: int,
    n_hidden_layers: int,
    lr: float,
    random_state: int = 0,
):
    n_labels = len(np.unique(y))
    device = get_device()
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
        # embeddings = clf.embedder(X_pt)
        # logits = clf.classifier(embeddings)
        logits = clf(X_pt)
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

    return clf, optimizer


def compute_gradients(
    X: np.ndarray,
    y: np.ndarray,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    n_labels: int,
) -> Tuple[np.ndarray, torch.Tensor, torch.Tensor]:
    device = get_device()
    X_pt = torch.tensor(X, dtype=torch.float32).to(device)
    y_pt = torch.tensor(y, dtype=torch.long).to(device)
    # now we are done training. Need to collect the grand scores here.
    model.eval()
    optimizer.zero_grad()
    embeddings = model.embedder(X_pt)
    logits = model.classifier(embeddings)
    loss = F.cross_entropy(logits.requires_grad_(True), y_pt)
    with torch.no_grad():
        param_grads = torch.autograd.grad(loss, logits)[0]
        gradients = torch.cat(
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
        )
    return (
        gradients.detach(),
        logits.detach(),
        embeddings.detach(),
    )


def lazy_greedy_optimize(
    X: np.ndarray,
    y: np.ndarray,
    clf: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    n_labels: int,
    budget: int,
):
    gradients, _, _ = compute_gradients(
        X=X,
        y=y,
        model=clf,
        optimizer=optimizer,
        n_labels=n_labels,
    )
    gradients = gradients.detach().cpu().numpy()
    lam = 1.0
    sim_matrix = np.zeros([X.shape[0], X.shape[0]], dtype=np.float32)
    sim_matrix_cols_sum = np.zeros(X.shape[0], dtype=np.float32)
    if_columns_calculated = np.zeros(X.shape[0], dtype=bool)
    all_idx = np.arange(X.shape[0])

    def similarity_kernel(a, b):
        if not np.all(if_columns_calculated[b]):
            if b.dtype != bool:
                temp = ~all_idx
                temp[b] = True
                b = temp
            not_calculated = b & ~if_columns_calculated
            sim_matrix[:, not_calculated] = cossim_np(
                gradients[all_idx],
                gradients[not_calculated],
            )
            sim_matrix_cols_sum[not_calculated] = np.sum(
                sim_matrix[:, not_calculated], axis=0
            )
            if_columns_calculated[not_calculated] = True
        return sim_matrix[np.ix_(a, b)]

    def gain_function(idx_gain):
        gain = (
            -2.0
            * np.sum(
                similarity_kernel(selected, idx_gain),
                axis=0,
            )
            + lam * sim_matrix_cols_sum[idx_gain]
        )
        return gain

    selected = np.zeros(len(X), dtype=bool)
    greedy_gain = np.zeros(len(X))
    greedy_gain[~selected] = gain_function(~selected)
    greedy_gain[selected] = -np.inf

    for _ in range(sum(selected), budget):
        best_gain = -np.inf
        last_max_element = -1
        while True:
            cur_max_element = greedy_gain.argmax()
            if last_max_element == cur_max_element:
                # Select cur_max_element into the current subset
                selected[cur_max_element] = True
                greedy_gain[cur_max_element] = -np.inf
                break
            new_gain = gain_function(np.array([cur_max_element]))[0]
            greedy_gain[cur_max_element] = new_gain
            if new_gain >= best_gain:
                best_gain = new_gain
                last_max_element = cur_max_element
    return selected


def graph_cut(
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

    sampled_X, sampled_y = [], []
    y_unique, y_counts = np.unique(y, return_counts=True)
    clf, optimizer = train_mlp(
        X=X,
        y=y,
        n_epochs=n_epochs,
        mlp_dim=mlp_dim,
        n_hidden_layers=n_hidden_layers,
        lr=lr,
        random_state=random_state,
    )

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
        selected = lazy_greedy_optimize(
            X=X[y == label],
            y=y[y == label],
            clf=clf,
            optimizer=optimizer,
            n_labels=len(y_unique),
            budget=ss,
        )
        sampled_X.append(X[y == label][selected])
        sampled_y += [label] * ss

    sampled_X = np.vstack(sampled_X)
    sampled_y = np.array(sampled_y)

    return sampled_X, sampled_y
