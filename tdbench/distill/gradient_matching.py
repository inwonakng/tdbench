import torch
from torch import nn, optim
import numpy as np
import copy
from functools import partial

from .random_sample import random_sample

# xavier uniform with controlled random generator
def xavier_uniform(tensor, gain=1.0, generator=None):
    fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(tensor)
    std = gain * np.sqrt(2.0 / float(fan_in + fan_out))
    a = np.sqrt(3.0) * std
    with torch.no_grad():
        return tensor.uniform_(-a, a, generator=generator)

def init_weights(m, gain=1.0, generator=None):
    if isinstance(m, nn.Linear):
        xavier_uniform(m.weight, gain=gain, generator=generator)
        m.bias.data.fill_(0.01)

def gradient_matching(
    X: np.ndarray,
    y: np.ndarray,
    N: int,
    n_epochs: int,
    mlp_dim: int,
    lr_mlp: float,
    lr_data: float,
    mom_data: float,
    n_hidden_layers: int,
    random_state: int | np.random.Generator = 0,
):
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"

    # to avoid complications, just generate randomly on cpu and then move to gpu.
    gen = torch.Generator("cpu")
    gen.manual_seed(random_state)
    n_labels = np.unique(y).shape[0]
    X_real = torch.tensor(X).float().to(device)
    y_real = torch.tensor(y).long().to(device)
    # X_syn = torch.randn(N*n_labels, X.shape[1], generator=gen).to(device)
    # y_syn = torch.tensor([i % n_labels for i in range(N*n_labels)]).to(device)

    idxs = np.arange(len(X))
    support_idxs, _ = random_sample(idxs, y, N, random_state=random_state)
    support_idxs = support_idxs.flatten()
    X_syn = torch.tensor(X[support_idxs]).float().to(device)
    y_syn = torch.tensor(y[support_idxs]).long().to(device)

    initializer = partial(init_weights, gain=1.0, generator=gen)

    model = nn.Sequential(
        nn.Linear(X.shape[1], mlp_dim),
        nn.ReLU(inplace=True),
        *[
            nn.Sequential(
                nn.Linear(mlp_dim, mlp_dim),
                nn.ReLU(inplace=True),
            )
            for _ in range(n_hidden_layers)
        ],
        nn.Linear(mlp_dim, n_labels),
        nn.Softmax(dim=1),
    ).apply(initializer).to(device)

    opt_model = optim.SGD(model.parameters(), lr=lr_mlp)
    opt_model.zero_grad()

    opt_data = torch.optim.SGD(
        [X_syn],
        lr=lr_data,
        momentum=mom_data,
    )  # optimizer_img for synthetic data
    opt_data.zero_grad()

    criterion = nn.CrossEntropyLoss()

    for epoch in range(n_epochs):
        loss_gw = torch.tensor(0.0).to(device)
        for label in np.unique(y):
            o_syn = model(X_syn[y_syn == label])
            o_real = model(X_real[y_real == label])

            l_syn = criterion(o_syn, y_syn[y_syn == label])
            l_real = criterion(o_real, y_real[y_real == label])

            gw_syn = torch.autograd.grad(l_syn, model.parameters(), create_graph=True)
            gw_real = (
                g.detach().clone()
                for g in torch.autograd.grad(l_real, model.parameters())
            )
            loss_gw += sum(
                # module-wise distance sum
                torch.sum(
                    1
                    - (
                        torch.sum(gwr * gws, dim=-1)
                        / (torch.norm(gwr, dim=-1) * torch.norm(gws, dim=-1) + 0.000001)
                    )
                )
                for gwr, gws in zip(gw_real, gw_syn)
            )
        opt_data.zero_grad()
        loss_gw.backward()
        opt_data.step()
        # print("match loss:", loss_gw.item())

        # update model with updated synthetic data
        X_syn_frozen, y_syn_frozen = copy.deepcopy(X_syn), copy.deepcopy(y_syn)
        out = model(X_syn_frozen)
        loss_model = criterion(out, y_syn_frozen)
        opt_model.zero_grad()
        loss_model.backward()
        opt_model.step()

        # print("model loss:", loss_model.item())
    distilled_X = X_syn_frozen.cpu().detach().numpy()
    distilled_y = y_syn_frozen.cpu().detach().numpy()
    return distilled_X, distilled_y
