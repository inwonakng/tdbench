from typing import Literal

import torch
import pytorch_lightning as pl

from ..utils import (
    get_optimizer,
    get_schedulers,
    step_scheduler,
    get_criterion,
    metric,
)

from .base_encoder import BaseEncoder
from .base_head import BaseHead

"""
Base encoder class for various tasks.

Has one encoder head and can have multiple decoder/downstream heads.
"""


class TabularAutoEncoder(pl.LightningModule):
    def __init__(
        self,
        name: str,
        encoder: BaseEncoder,
        heads: list[BaseHead],
        loss_balance: list[float],
        metrics: list[list[str]],
        in_dim: int,
        feature_mask: tuple[int] | list[int] | torch.LongTensor,
        feature_categ_mask: tuple[bool] | list[bool] | torch.BoolTensor,
        opt_name: str,
        opt_lr: float,
        opt_wd: float,
        sch_names: tuple[str] | list[str] = (),
        **kwargs,
    ) -> None:
        super().__init__()

        self.name = name
        self.encoder = encoder
        self.heads = heads
        self.tasks = tasks
        self.criterion = get_criterion(criterion)
        self.opt_name = opt_name
        self.opt_lr = opt_lr
        self.opt_wd = opt_wd
        self.sch_names = sch_names
        self.metrics = metrics
        self.in_dim = in_dim
        self.loss_balance = loss_balance

        # These are loaded from file because they can be very big and ray tune will save everything..
        self.feature_mask = (
            feature_mask
            if isinstance(feature_mask, torch.Tensor)
            else torch.tensor(feature_mask).long()
        )
        self.feature_categ_mask = (
            feature_categ_mask
            if isinstance(feature_categ_mask, torch.Tensor)
            else torch.tensor(feature_categ_mask).bool()
        )

        self._loss = dict(train=[], val=[], test=[])

        self._target_true = [dict(train=[], val=[], test=[]) for _ in range(len(heads))]
        self._target_pred = [dict(train=[], val=[], test=[]) for _ in range(len(heads))]

        self.automatic_optimization = False

    def evaluate(self, stage_tag: Literal["train", "val", "test"], **kwargs) -> None:
        """Executes the common steps to calculate the average loss and metrics.
        Also tests the tester models' predictive metrics.

        Args:
            stage_tag (Literal['train', 'val', 'test']): Tag of the stage evaluation is taking place in.
        """

        for i, (t, h) in enumerate(zip(self.tasks, self.heads)):
            ...

        target_pred = torch.vstack(self._target_pred[stage_tag])
        target_true = torch.vstack(self._target_true[stage_tag])
        self.log_dict(
            {
                f"{stage_tag}/loss": sum(self._loss[stage_tag])
                / len(self._loss[stage_tag]),
                **{
                    f"{stage_tag}/{m}": metric(
                        m,
                        target_true,
                        target_pred,
                        feature_mask=self.feature_mask,
                        feature_categ_mask=self.feature_categ_mask,
                    )
                    for m in self.metrics
                },
            },
            sync_dist=True,
        )

        self._loss[stage_tag].clear()
        self._target_true[stage_tag].clear()
        self._target_pred[stage_tag].clear()

    def compute_task(
        self,
        stage_tag: str,
        batch: tuple[torch.FloatTensor, torch.LongTensor, torch.LongTensor],
    ) -> torch.FloatTensor:
        """

        Args:
            batch (tuple[torch.FloatTensor, torch.LongTensor, torch.LongTensor]):
                tuple of `X (batch_size, num_col)`,
                `y (batch_size)` and `feature_idx (num_col)`

        Returns:
            torch.FloatTensor: `loss (1)`
        """

        x, y, feature_idx = batch

        for i, (t, h, c) in enumerate(zip(self.tasks, self.heads, self.criterion)):
            if t == "reconstruct":
                encoded = self.encode(x=x, feature_idx=feature_idx)
                out = self.decode(encoded)
                loss = c(
                    input=out,
                    target=x,
                    feature_mask=self.feature_mask,
                    feature_categ_mask=self.feature_categ_mask,
                )
                ...
            elif t == "classify":
                ...
            else:
                raise NotImplementedError

        encoded = self.encode(x=x, feature_idx=feature_idx)
        out = self.decode(encoded)
        loss = self.criterion(
            input=out,
            target=x,
            feature_mask=self.feature_mask,
            feature_categ_mask=self.feature_categ_mask,
        )

        self._loss[stage_tag].append(loss.item())
        self._target_true[stage_tag].append(x)
        self._target_pred[stage_tag].append(out)
        return loss

    def training_step(
        self,
        batch,
        batch_idx,
    ) -> torch.FloatTensor:
        """Called at each batch of train_loader"""
        loss = self.compute_task("train", batch)
        opt = self.optimizers()
        opt.zero_grad()
        self.manual_backward(loss)
        opt.step()

        if self.trainer.is_last_batch:
            self.evaluate("train")
        return loss

    def validation_step(self, batch, batch_idx) -> torch.FloatTensor:
        """Called at each batch of val_loader"""
        loss = self.compute_task("val", batch)
        return loss

    def on_validation_epoch_end(self) -> torch.FloatTensor:
        self.evaluate("val")
        if len(self.sch_names) == 0:
            pass
        elif len(self.sch_names) == 1:
            sch = self.lr_schedulers()
            step_scheduler(sch, self.trainer.callback_metrics["val/loss"])
        else:
            schedulers = self.lr_schedulers()
            for sch in schedulers:
                step_scheduler(sch, self.trainer.callback_metrics["val/loss"])

    def test_step(self, batch, batch_idx) -> None:
        """Called at each batch of test_loader"""
        loss = self.compute_task("test", batch)
        return loss

    def on_test_epoch_end(self) -> None:
        self.evaluate("test")

    def configure_optimizers(self) -> tuple[list[torch.optim.Optimizer], list[dict]]:
        """Configures the optimizers and lr_schedulers.

        Returns:
            tuple[list[torch.optim.Optimizer], list[dict]]: List of optimizers and a list of configurations for the corresponding schedulers.
        """

        opt = get_optimizer(
            self.parameters(),
            opt_name=self.opt_name,
            lr=self.opt_lr,
            weight_decay=self.opt_wd,
        )
        optimizers = [opt]
        if self.sch_names:
            schedulers = get_schedulers(
                optimizer=opt,
                sch_names=self.sch_names,
            )
        else:
            schedulers = []
        return optimizers, schedulers

    def encode(self, x, **kwargs):
        ...

    def decode(self, encoded, **kwargs):
        ...

    def convert_binary(
        self,
        decoded_cont: torch.FloatTensor,
    ) -> torch.LongTensor:
        """Converts the decoder output in continuous space into a binary one-hot encoding form.

        Args:
            decoded_cont (torch.FloatTensor): _description_

        Returns:
            torch.LongTensor: _description_
        """
        decoded = []
        n_features = self.feature_mask.max() + 1
        for i in range(n_features):
            mask = (self.feature_mask == i).nonzero().squeeze(1)
            # only keep the max value as 1 and everything else as 0
            converted = (
                decoded_cont[:, mask] == decoded_cont[:, mask].amax(1).unsqueeze(1)
            ).long()
            decoded.append(converted)
        decoded = torch.hstack(decoded)
        return decoded
