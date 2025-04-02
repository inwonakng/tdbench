"""
Code from https://gzyaftermath.github.io/DATM/
Mostly the same code as https://github.com/GeorgeCazenavette/mtt-distillation with some additional strategy for
trajectory selection

"""

import torch
from torch import nn, optim
import numpy as np
import random
import torch.nn.functional as F

from .random_sample import random_sample
from .utils import get_mlp, ReparamModule


def SoftCrossEntropy(inputs, target, reduction="average"):
    input_log_likelihood = -F.log_softmax(inputs, dim=1)
    target_log_likelihood = F.softmax(target, dim=1)
    batch = inputs.shape[0]
    loss = torch.sum(torch.mul(input_log_likelihood, target_log_likelihood)) / batch
    return loss


"""
Get trajectories of expert model
"""


def get_trajectories(
    X: np.ndarray,
    y: np.ndarray,
    n_experts: int,
    n_epochs: int,
    mlp_dim: int,
    n_hidden_layers: int,
    lr_mlp: float,
    random_state: int = 0,
):
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"

    n_labels = np.unique(y).shape[0]
    X_pt = torch.tensor(X).float().to(device)
    y_pt = torch.tensor(y).long().to(device)

    expert_seeds = list(range(1000))
    rng = random.Random(random_state)
    rng.shuffle(expert_seeds)

    all_trajectories = []
    for i in range(n_experts):
        trajectories = []
        random_state = expert_seeds[0]
        model = get_mlp(
            input_shape=X.shape[1],
            n_hidden_layers=n_hidden_layers,
            hidden_dim=mlp_dim,
            n_labels=n_labels,
            random_state=random_state,
        ).to(device)

        trajectories.append([p.detach().cpu() for p in model.parameters()])
        opt_model = optim.SGD(model.parameters(), lr=lr_mlp)
        opt_model.zero_grad()

        criterion = nn.CrossEntropyLoss()
        for epoch in range(n_epochs):
            opt_model.zero_grad()
            out = model(X_pt)
            loss = criterion(out, y_pt)
            loss.backward()
            opt_model.step()
            trajectories.append([p.detach().cpu() for p in model.parameters()])
        all_trajectories.append(trajectories)
    return all_trajectories


def datm(
    X: np.ndarray,
    y: np.ndarray,
    N: int,
    n_epochs: int,
    n_experts: int,
    expert_epochs: int,
    syn_steps: int,
    mlp_dim: int,
    n_iter: int,
    lr_teacher: float,
    lr_data: float,
    lr_lr: float,
    mom_lr: float,
    mom_data: float,
    min_start_epoch: int,
    current_max_start_epoch: int,
    max_start_epoch: int,
    expansion_end_epoch: int,
    n_hidden_layers: int,
    random_state: int | np.random.Generator = 0,
):
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"

    trajectories = get_trajectories(
        X=X,
        y=y,
        n_experts=n_experts,
        n_epochs=n_epochs,
        mlp_dim=mlp_dim,
        n_hidden_layers=n_hidden_layers,
        lr_mlp=lr_teacher,
        random_state=random_state,
    )

    n_labels = np.unique(y).shape[0]
    idxs = np.arange(len(X))
    support_idxs, _ = random_sample(idxs, y, N, random_state=random_state)
    support_idxs = support_idxs.flatten()
    X_syn = torch.tensor(X[support_idxs]).float().to(device)
    y_syn = torch.tensor(y[support_idxs]).long().to(device)
    param_loss_list = []
    param_dist_list = []
    syn_lr = torch.tensor(lr_teacher).to(device)
    optimizer_data = torch.optim.SGD([X_syn], lr=lr_data, momentum=mom_data)
    optimizer_lr = torch.optim.SGD([syn_lr], lr=lr_lr, momentum=mom_lr)
    optimizer_data.zero_grad()

    param_cache = {}

    for it in range(n_iter):
        student_net = ReparamModule(
            get_mlp(
                input_shape=X.shape[1],
                n_hidden_layers=n_hidden_layers,
                hidden_dim=mlp_dim,
                n_labels=n_labels,
                random_state=random_state,
            )
        ).to(device)
        student_net.train()
        num_params = sum([np.prod(p.size()) for p in (student_net.parameters())])

        rng = random.Random(random_state)

        # This is where the difference from mtt is.
        upper_bound = current_max_start_epoch + int(
            (max_start_epoch - current_max_start_epoch) * it / expansion_end_epoch
        )
        upper_bound = min(upper_bound, max_start_epoch)
        start_epoch = rng.randint(min_start_epoch, upper_bound)

        expert_idx = rng.randint(0, n_experts - 1)
        starting_params = trajectories[expert_idx][start_epoch]
        target_params = trajectories[expert_idx][start_epoch + expert_epochs]

        if (expert_idx, start_epoch) in param_cache:
            starting_params = param_cache[(expert_idx, start_epoch)]
        else:
            starting_params = trajectories[expert_idx][start_epoch]
            param_cache[(expert_idx, start_epoch)] = starting_params

        if (expert_idx, start_epoch + expert_epochs) in param_cache:
            target_params = param_cache[(expert_idx, start_epoch + expert_epochs)]
        else:
            target_params = trajectories[expert_idx][start_epoch + expert_epochs]
            param_cache[(expert_idx, start_epoch + expert_epochs)] = target_params

        target_params = torch.cat(
            [p.data.to(device).reshape(-1) for p in target_params], 0
        )

        student_params = [
            torch.cat(
                [p.data.to(device).reshape(-1) for p in starting_params], 0
            ).requires_grad_(True)
        ]

        starting_params = torch.cat(
            [p.data.to(device).reshape(-1) for p in starting_params], 0
        )

        for step in range(syn_steps):
            forward_params = student_params[-1]
            y_pred = student_net(X_syn, flat_param=forward_params)
            ce_loss = F.cross_entropy(y_pred, y_syn)
            grad = torch.autograd.grad(
                ce_loss,
                student_params[-1],
                create_graph=True,
            )[0]
            student_params.append(student_params[-1] - syn_lr * grad)

        param_loss = torch.tensor(0.0).to(device)
        param_dist = torch.tensor(0.0).to(device)

        param_loss += F.mse_loss(student_params[-1], target_params, reduction="sum")
        param_dist += F.mse_loss(starting_params, target_params, reduction="sum")

        param_loss_list.append(param_loss)
        param_dist_list.append(param_dist)

        param_loss /= num_params
        param_dist /= num_params

        param_loss /= param_dist

        grand_loss = param_loss

        optimizer_data.zero_grad()
        optimizer_lr.zero_grad()

        grand_loss.backward()

        optimizer_data.step()
        optimizer_lr.step()

        for _ in student_params:
            del _

    distilled_X = X_syn.cpu().detach().numpy()
    distilled_y = y_syn.cpu().detach().numpy()

    return distilled_X, distilled_y
