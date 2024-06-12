from typing import Literal

import torch
import pytorch_lightning as pl

from .utils import (
    get_optimizer,
    get_schedulers,
    step_scheduler,
    get_criterion,
    metric,
)


class BaseModel(pl.LightningModule):
    name: str

    def __init__(
        self,
        # config: dict,
        criterion: str,
        metrics: list[str],
        feature_mask: tuple[int] | list[int] | torch.LongTensor,
        feature_categ_mask: tuple[bool] | list[bool] | torch.BoolTensor,
        opt_name: str,
        opt_lr: float,
        opt_wd: float,
        sch_names: tuple[str] | list[str] = (),
        **kwargs,
    ) -> None:
        super().__init__()
        self.criterion = get_criterion(criterion)
        self.opt_name = opt_name
        self.opt_lr = opt_lr
        self.opt_wd = opt_wd
        self.sch_names = sch_names
        self.metrics = metrics
        self.num_features = feature_mask.max() + 1
        self.feature_mask = feature_mask
        self.feature_categ_mask = feature_categ_mask

        # storage for prediction/true labels.
        # better to do this manually because
        # metrics like balanced accuracy don't play nice with just weighted averaging
        self._target_true = {"train": [], "val": [], "test": []}
        self._target_pred = {"train": [], "val": [], "test": []}
        self._loss = {"train": [], "val": [], "test": []}

        self.automatic_optimization = False

    def evaluate(self, stage_tag: Literal["train", "val", "test"], **kwargs) -> None:
        """Executes the common steps to calculate the average loss and metrics.

        Args:
            stage_tag (Literal['train', 'val', 'test']): Tag of the stage evaluation is taking place in.
        """
        target_pred = torch.hstack(self._target_pred[stage_tag])
        target_true = torch.hstack(self._target_true[stage_tag])
        eval_results = {
            f"{stage_tag}/loss": sum(self._loss[stage_tag])
            / len(self._loss[stage_tag]),
            **{
                f"{stage_tag}/{m}": metric(
                    metric_name=m,
                    target_true=target_true.detach().cpu(),
                    target_pred=target_pred.detach().cpu(),
                    feature_mask=self.feature_mask,
                    feature_categ_mask=self.feature_categ_mask,
                    num_classes=self.out_dim,
                )
                for m in self.metrics
            },
        }
        print("We log these", eval_results.keys())
        self.log_dict(eval_results)

        self._loss[stage_tag].clear()
        self._target_true[stage_tag].clear()
        self._target_pred[stage_tag].clear()

    def compute_task(
        self,
        stage_tag: str,
        batch: tuple[torch.FloatTensor, torch.LongTensor, torch.LongTensor],
    ) -> torch.FloatTensor:
        """_summary_

        Args:
            batch (tuple[torch.FloatTensor, torch.LongTensor, torch.LongTensor]):
                tuple of `X (batch_size, num_col)`,
                `y (batch_size)` and `feature_idx (num_col)`

        Returns:
            torch.FloatTensor: `loss (1)s`
        """
        x, y, feature_idx = batch
        out = self.forward(x=x, feature_idx=feature_idx)
        loss = self.criterion(
            input=out,
            target=y,
            feature_mask=self.feature_mask,
            feature_categ_mask=self.feature_categ_mask,
        )
        true = y
        pred = out.argmax(1)

        self._loss[stage_tag].append(loss)
        self._target_true[stage_tag].append(true)
        self._target_pred[stage_tag].append(pred)

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

    def on_validation_epoch_end(self) -> None:
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

    def test_step(self, batch, batch_idx) -> torch.FloatTensor:
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

