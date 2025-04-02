import torch
import numpy as np
from rtdl_revisiting_models import FTTransformer
from sklearn.base import BaseEstimator, ClassifierMixin

from tabdd.data import TabularDataset
from ..utils import (
    get_optimizer,
    is_onehot,
)

default_args = dict(
    n_blocks=3,
    d_block=192,
    attention_n_heads=8,
    attention_dropout=0.2,
    ffn_d_hidden=None,
    ffn_d_hidden_multiplier=4 / 3,
    ffn_dropout=0.1,
    residual_dropout=0.0,
)


class ScikitFTTransformer(BaseEstimator, ClassifierMixin):
    model: FTTransformer
    params: dict
    optimizer: torch.optim.Optimizer
    categ_features: np.ndarray
    features: np.ndarray
    feature_counts: np.ndarray

    d_block: int
    n_blocks: int
    attention_n_heads: int
    attention_dropout: float
    ffn_d_hidden_multiplier: float
    ffn_dropout: float
    residual_dropout: float
    sample_dset: TabularDataset
    early_stopping: bool = True
    max_epochs: int = 1000
    patience: int = 16

    opt_name: str
    opt_lr: float
    opt_wd: float

    is_onehot: bool

    classes_: int
    device: str

    def __init__(
        self,
        d_block: int,
        n_blocks: int,
        attention_n_heads: int,
        attention_dropout: float,
        ffn_d_hidden_multiplier: float,
        ffn_dropout: float,
        residual_dropout: float,
        opt_name: str,
        opt_lr: float,
        opt_wd: float,
        sample_dset: TabularDataset,
        early_stopping: bool = True,
        max_epochs: int = 1000,
        patience: int = 16,
        device: str = "cpu",
    ) -> None:
        self.d_block = d_block
        self.n_blocks = n_blocks
        self.attention_n_heads = attention_n_heads
        self.attention_dropout = attention_dropout
        self.ffn_d_hidden_multiplier = ffn_d_hidden_multiplier
        self.ffn_dropout = ffn_dropout
        self.residual_dropout = residual_dropout
        self.opt_name = opt_name
        self.opt_lr = opt_lr
        self.opt_wd = opt_wd
        self.sample_dset = sample_dset
        self.early_stopping = early_stopping
        self.max_epochs = max_epochs
        self.patience = patience

        self.is_onehot = is_onehot(sample_dset.X)

        self.features, self.feature_counts = np.unique(
            self.sample_dset.feature_mask, return_counts=True
        )

        # determining the cardinality of categorical features
        # if the dataset is already one-hot, everything is categorical. just need to look at feature_mask in that case.
        # if not, we need to look at feature_categ_mask
        if self.is_onehot:
            self.cat_cardinalities = np.unique(
                sample_dset.feature_mask,
                return_counts=True,
            )[1].tolist()
            self.categ_features = np.ones_like(sample_dset.feature_mask, dtype=bool)
            self.n_cont_features = 0
            self.is_cat = np.ones_like(self.features, dtype=bool)
        else:
            self.cat_cardinalities = [
                sum(sample_dset.feature_mask == i)
                for i, is_cat in enumerate(sample_dset.feature_categ_mask)
                if is_cat
            ]

            self.categ_features = np.array(
                [sample_dset.feature_categ_mask[f] for f in sample_dset.feature_mask]
            )

            self.n_cont_features = len(sample_dset.feature_mask) - sum(
                self.cat_cardinalities
            )
            self.is_cat = sample_dset.feature_categ_mask

        d_out = np.unique(sample_dset.y).shape[0]
        self.classes_ = np.arange(d_out)
        self.device = device

        self.model = FTTransformer(
            n_cont_features=self.n_cont_features,
            cat_cardinalities=self.cat_cardinalities,
            d_out=d_out,
            d_block=d_block,
            n_blocks=n_blocks,
            attention_n_heads=attention_n_heads,
            attention_dropout=attention_dropout,
            ffn_d_hidden_multiplier=ffn_d_hidden_multiplier,
            ffn_dropout=ffn_dropout,
            residual_dropout=residual_dropout,
        ).to(self.device)

        self.optimizer = get_optimizer(
            self.model.parameters(),
            opt_name=opt_name,
            lr=opt_lr,
            weight_decay=opt_wd,
        )

    def parse_data(
        self, X: np.ndarray
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:

        if (~self.categ_features).any():
            cont = X[:, ~self.categ_features]
            if not isinstance(cont, torch.Tensor):
                cont = torch.tensor(cont).float()
            else:
                cont = cont.float()
        else:
            cont = None

        # FTTransformer expects categorical features to be nominal.
        # So we need to convert it back by taking the argmax at each feature group.
        if self.categ_features.any():
            cat = torch.tensor(
                np.hstack(
                    [
                        X[:, self.sample_dset.feature_mask == f].argmax(1)[:, None]
                        for f in self.features
                        if self.is_cat[f]
                    ]
                )
            ).long()
        else:
            cat = None

        return cont, cat

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> None:
        cont, cat = self.parse_data(X)
        if cont is not None:
            cont = cont.to(self.device)
        if cat is not None:
            cat = cat.to(self.device)
        loss_fn = torch.nn.CrossEntropyLoss()
        best_loss = 1_000_000
        n_bad_conseq_updates = 0

        y_ = y
        if not isinstance(y_, torch.Tensor):
            y_ = torch.tensor(y_).long()
        y_ = y_.to(self.device)

        self.model.train()
        for epoch in range(self.max_epochs):
            self.optimizer.zero_grad()
            out = self.model(cont, cat)
            loss = loss_fn(out, y_)
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
        cont, cat = self.parse_data(X)
        if cont is not None:
            cont = cont.to(self.device)
        if cat is not None:
            cat = cat.to(self.device)
        pred = self.model(cont, cat).argmax(1).detach().cpu().numpy().astype(int)
        return pred
