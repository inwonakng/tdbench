from typing import Literal

import torch
import pytorch_lightning as pl

from .gnn_autoencoder import GNNAutoEncoder
from .mlp_autoencoder import MLPAutoEncoder
from ..utils import (
    get_optimizer,
    get_schedulers,
    step_scheduler,
    get_criterion,
    metric,
)

# from .scheduler import get_schedulers, step_scheduler
# from .criterion import get_criterion
# from .metric import metric
from ..modules import MLPModule


class MultiHeadAutoEncoder(pl.LightningModule):
    def __init__(
        self,
        autoencoder: GNNAutoEncoder | MLPAutoEncoder,
        in_dim: int,
        out_dim: int,
        latent_dim: int,
        feature_mask: tuple[int] | list[int] | torch.LongTensor,
        feature_categ_mask: tuple[bool] | list[bool] | torch.BoolTensor,
        classifier_hidden_dims: tuple[int] | list[int],
        classifier_dropout_p: float,
        opt_name: str,
        opt_lr: float,
        opt_wd: float,
        autoencoder_metrics: list[str],
        classifier_metrics: list[str],
        autoencoder_criterion: str,
        classifier_criterion: str,
        combined_loss_balance: float,
        combined_metric_balance: float,
        sch_names: tuple[str] | list[str] = (),
        **kwargs,
    ) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.latent_dim = latent_dim

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

        self.autoencoder_criterion = get_criterion(autoencoder_criterion)
        self.classifier_criterion = get_criterion(classifier_criterion)
        self.opt_name = opt_name
        self.opt_lr = opt_lr
        self.opt_wd = opt_wd
        self.sch_names = sch_names

        self.autoencoder_metrics = autoencoder_metrics
        self.classifier_metrics = classifier_metrics
        self.combined_loss_balance = combined_loss_balance
        self.combined_metric_balance = combined_metric_balance

        assert (
            (self.in_dim == autoencoder.in_dim)
            and (self.latent_dim == autoencoder.latent_dim)
            and all(self.feature_mask == autoencoder.feature_mask)
            and all(self.feature_categ_mask == autoencoder.feature_categ_mask)
        )

        self.autoencoder = autoencoder

        # First ensure that we can use this autoencoder
        self.classifier = MLPModule(
            in_dim=self.latent_dim,
            out_dim=out_dim,
            hidden_dims=list(classifier_hidden_dims),
            dropout_p=classifier_dropout_p,
            use_batchnorm=True,
        )

        self._loss = dict(train=[], val=[], test=[])
        self._autoencoder_target_true = dict(train=[], val=[], test=[])
        self._autoencoder_target_pred = dict(train=[], val=[], test=[])
        self._classifier_target_true = dict(train=[], val=[], test=[])
        self._classifier_target_pred = dict(train=[], val=[], test=[])

        self.automatic_optimization = False

    @property
    def name(self):
        return "MultiHead" + self.autoencoder.name

    def evaluate(self, stage_tag: Literal["train", "val", "test"], **kwargs) -> None:
        """Executes the common steps to calculate the average loss and metrics.
        Also tests the tester models' predictive metrics.


        Args:
            stage_tag (Literal['train', 'val', 'test']): Tag of the stage evaluation is taking place in.
        """

        autoencoder_target_true = torch.vstack(self._autoencoder_target_true[stage_tag])
        autoencoder_target_pred = torch.vstack(self._autoencoder_target_pred[stage_tag])
        classifier_target_true = torch.hstack(self._classifier_target_true[stage_tag])
        classifier_target_pred = torch.hstack(self._classifier_target_pred[stage_tag])

        autoencoder_scores = {
            f"{stage_tag}/{m}": metric(
                m,
                autoencoder_target_true,
                autoencoder_target_pred,
                feature_mask=self.feature_mask,
                feature_categ_mask=self.feature_categ_mask,
            )
            for m in self.autoencoder_metrics
        }
        classifier_scores = {
            f"{stage_tag}/{m}": metric(
                m,
                classifier_target_true,
                classifier_target_pred,
                num_classes=self.out_dim,
            )
            for m in self.classifier_metrics
        }
        self.log_dict(
            {
                f"{stage_tag}/loss": sum(self._loss[stage_tag])
                / len(self._loss[stage_tag]),
                f"{stage_tag}/combined_score": (
                    list(autoencoder_scores.values())[0] * self.combined_metric_balance
                    + list(classifier_scores.values())[0]
                    * (1 - self.combined_metric_balance)
                ),
                **autoencoder_scores,
                **classifier_scores,
            },
            sync_dist=True,
        )

        self._loss[stage_tag].clear()
        self._autoencoder_target_true[stage_tag].clear()
        self._autoencoder_target_pred[stage_tag].clear()
        self._classifier_target_true[stage_tag].clear()
        self._classifier_target_pred[stage_tag].clear()

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
        encoded = self.encode(
            x=x,
            feature_idx=feature_idx,
            feature_mask=self.feature_mask.to(x.device),
            feature_categ_mask=self.feature_categ_mask.to(x.device),
        )
        autoencoder_out = self.decode(encoded)
        autoencoder_loss = self.autoencoder_criterion(
            input=autoencoder_out,
            target=x,
            feature_mask=self.feature_mask,
            feature_categ_mask=self.feature_categ_mask,
        )
        classifier_out = self.classifier(encoded)
        classifier_loss = self.classifier_criterion(
            input=classifier_out,
            target=y,
        )

        loss = autoencoder_loss * self.combined_loss_balance + classifier_loss * (
            1 - self.combined_loss_balance
        )

        self._loss[stage_tag].append(loss.item())
        self._autoencoder_target_true[stage_tag].append(x)
        self._autoencoder_target_pred[stage_tag].append(autoencoder_out)
        self._classifier_target_true[stage_tag].append(y)
        self._classifier_target_pred[stage_tag].append(classifier_out.argmax(1))
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

    def validation_step(self, batch, batch_idx) -> None:
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
        return self.autoencoder.encode(x, **kwargs)

    def decode(self, encoded, **kwargs):
        return self.autoencoder.decode(encoded)

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
