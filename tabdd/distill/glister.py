from typing import Tuple, Optional
import numpy as np

import torch.nn.functional as F
import torch
from .utils import get_mlp


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

    return (
        clf,
        optimizer,
    )


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
    eta: float,
    n_labels: int,
    budget: int,
):
    device = get_device()
    gradients, init_out, init_emb = compute_gradients(
        X=X,
        y=y,
        model=clf,
        optimizer=optimizer,
        n_labels=n_labels,
    )
    train_grads = gradients.clone()
    val_grads = gradients.mean(dim=0).clone()
    y_pt = torch.tensor(y, dtype=torch.long).to(device)

    def gain_function(idx_gain):
        return (
            torch.matmul(
                train_grads[idx_gain],
                val_grads.view(-1, 1),
            )
            .detach()
            .cpu()
            .numpy()
            .flatten()
        )

    def recompute_val_grad(selected_for_train):
        sum_selected_train_gradients = train_grads[selected_for_train].mean(dim=0)
        new_outputs = (
            init_out
            - eta
            * sum_selected_train_gradients[:n_labels]
            .view(1, -1)
            .repeat(init_out.shape[0], 1)
            - eta
            * torch.matmul(
                init_emb,
                sum_selected_train_gradients[n_labels:].view(n_labels, -1).T,
            )
        )
        sample_num = new_outputs.shape[0]
        gradients = torch.zeros(
            [sample_num, n_labels * (init_emb.size(1) + 1)],
            requires_grad=False,
        ).to(device)
        batch_indx = np.arange(sample_num)
        new_outputs_batch = (
            new_outputs[batch_indx].clone().detach().requires_grad_(True)
        )
        loss = F.cross_entropy(
            new_outputs_batch,
            y_pt[batch_indx],
        )
        batch_num = len(batch_indx)
        bias_parameters_grads = torch.autograd.grad(
            loss.sum(),
            new_outputs_batch,
            retain_graph=True,
        )[0]
        weight_parameters_grads = init_emb[batch_indx].view(
            batch_num, 1, init_emb.size(1)
        ).repeat(1, n_labels, 1) * bias_parameters_grads.view(
            batch_num, n_labels, 1
        ).repeat(
            1, 1, init_emb.size(1)
        )
        gradients[batch_indx] = torch.cat(
            [
                bias_parameters_grads,
                weight_parameters_grads.flatten(1),
            ],
            dim=1,
        )
        return gradients.mean(dim=0)

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
                val_grads = recompute_val_grad(selected)
                break
            new_gain = gain_function(np.array([cur_max_element]))[0]
            greedy_gain[cur_max_element] = new_gain
            if new_gain >= best_gain:
                best_gain = new_gain
                last_max_element = cur_max_element
    return selected


def glister(
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
            eta=lr,
            n_labels=len(y_unique),
            budget=ss,
        )

        sampled_X.append(X[y == label][selected])
        sampled_y += [label] * ss

    sampled_X = np.vstack(sampled_X)
    sampled_y = np.array(sampled_y)

    return sampled_X, sampled_y
