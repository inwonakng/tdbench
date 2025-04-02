import torch
import numpy as np

from sklearn.base import BaseEstimator, ClassifierMixin
from rtdl_revisiting_models import ResNet

from tabdd.data import TabularDataset
from ..utils import (
    get_optimizer,
)


class ScikitResNet(BaseEstimator, ClassifierMixin):
    model: ResNet
    optimizer: torch.optim.Optimizer

    n_blocks: int
    d_block: int
    d_hidden_multiplier: float
    dropout1: float
    dropout2: float
    sample_dset: TabularDataset
    d_hidden: int

    opt_name: str
    opt_lr: float
    opt_wd: float

    early_stopping: bool
    max_epochs: int
    patience: int

    classes_: int
    device: str

    def __init__(
        self,
        n_blocks: int,
        d_block: int,
        d_hidden_multiplier: float,
        dropout1: float,
        dropout2: float,
        opt_name: str,
        opt_lr: float,
        opt_wd: float,
        sample_dset: TabularDataset,
        early_stopping: bool = True,
        max_epochs: int = 1000,
        patience: int = 16,
        device: str = "cpu",
    ) -> None:
        self.n_blocks = n_blocks
        self.d_block = d_block
        self.d_hidden_multiplier = d_hidden_multiplier
        self.dropout1 = dropout1
        self.dropout2 = dropout2
        self.opt_name = opt_name
        self.opt_lr = opt_lr
        self.opt_wd = opt_wd
        self.sample_dset = sample_dset
        self.early_stopping = early_stopping
        self.max_epochs = max_epochs
        self.patience = patience

        d_in = sample_dset.X.shape[1]
        d_out = np.unique(sample_dset.y).shape[0]
        self.classes_ = np.arange(d_out)
        self.device = device

        self.model = ResNet(
            d_in=d_in,
            d_out=d_out,
            n_blocks=n_blocks,
            d_block=d_block,
            d_hidden_multiplier=d_hidden_multiplier,
            dropout1=dropout1,
            dropout2=dropout2,
        ).to(self.device)

        self.optimizer = get_optimizer(
            self.model.parameters(),
            opt_name=opt_name,
            lr=opt_lr,
            weight_decay=opt_wd,
        )

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> None:
        loss_fn = torch.nn.CrossEntropyLoss()
        best_loss = 1_000_000
        n_bad_conseq_updates = 0

        X_pt = torch.tensor(X) if not isinstance(X, torch.Tensor) else X
        y_pt = torch.tensor(y) if not isinstance(y, torch.Tensor) else y
        X_pt = X_pt.to(self.device).float()
        y_pt = y_pt.to(self.device).long()

        self.model.train()
        for epoch in range(self.max_epochs):
            self.optimizer.zero_grad()
            out = self.model(X_pt)
            loss = loss_fn(out, y_pt)
            loss.backward()
            self.optimizer.step()

            if loss > best_loss:
                n_bad_conseq_updates += 1
            else:
                best_loss = loss
                n_bad_conseq_updates = 0
            if n_bad_conseq_updates > self.patience:
                break
        return self

    def predict(self, X) -> np.ndarray:
        self.model.eval()
        X_pt = torch.tensor(X).float() if not isinstance(X, torch.Tensor) else X.float()
        X_pt = X_pt.to(self.device)
        pred = self.model(X_pt).argmax(1).cpu().detach().numpy().astype(int)
        return pred
