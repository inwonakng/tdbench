from typing import Tuple, Optional
import pandas as pd
import torch
from pathlib import Path
import shutil
import json

from tabdd.config import (
    EncoderTuneConfig,
    MultiEncoderTuneConfig,
    EncoderTrainConfig,
)

from .gnn_autoencoder import GNNAutoEncoder
from .mlp_autoencoder import MLPAutoEncoder
from .tf_autoencoder import TFAutoEncoder
from .multihead_autoencoder import MultiHeadAutoEncoder


def get_best_checkpoint(
    best_checkpoint_dir: str | Path,
    metric: str = "val/recon_accuracy_score",
    direction: str = "max",
    clean_directory: bool = False,
) -> Optional[Path]:
    best_checkpoints = sorted(Path(best_checkpoint_dir).glob("*/metrics.csv"))
    if not best_checkpoints:
        return None
    results = [(pd.read_csv(f)[metric].values[-1], f) for f in best_checkpoints]
    if direction == "max":
        _, best_checkpoint = max(results)
    elif direction == "min":
        _, best_checkpoint = min(results)
    else:
        raise NotImplementedError

    if clean_directory:
        for _, f in results:
            if f != best_checkpoint:
                shutil.rmtree(f.parent)

    return best_checkpoint.parent


def load_encoder(
    encoder_tune_config: EncoderTuneConfig,
    encoder_train_config: EncoderTrainConfig,
    data_module: any,
    device: str | torch.device = "cpu",
) -> Tuple[
    Optional[GNNAutoEncoder | MLPAutoEncoder | MultiHeadAutoEncoder],
    Optional[Path],
]:
    if isinstance(encoder_tune_config, MultiEncoderTuneConfig):
        encoder_cls = MultiHeadAutoEncoder
        metric = "val/combined_score"
        base_autoencoder, _ = load_encoder(
            encoder_tune_config=encoder_tune_config.base_encoder_config,
            encoder_train_config=encoder_train_config,
            data_module=data_module,
            device=device,
        )
    elif encoder_tune_config.identifier == "gnnautoencoder":
        encoder_cls = GNNAutoEncoder
        metric = "val/recon_accuracy_score"
        base_autoencoder = None
    elif encoder_tune_config.identifier == "mlpautoencoder":
        encoder_cls = MLPAutoEncoder
        metric = "val/recon_accuracy_score"
        base_autoencoder = None
    elif encoder_tune_config.identifier == "tfautoencoder":
        encoder_cls = TFAutoEncoder
        metric = "val/recon_accuracy_score"
        base_autoencoder = None
    else:
        raise NotImplementedError

    encoder_checkpoint = (
        encoder_train_config.checkpoint_dir
        / encoder_tune_config.identifier
        / data_module.dataset_config.identifier
        / data_module.data_mode_config.identifier
        / "asha_hyperopt"
        / str(encoder_train_config.latent_dim)
    )

    best_checkpoint = get_best_checkpoint(encoder_checkpoint, metric)

    if best_checkpoint is not None:
        with open(best_checkpoint / "model_config.json") as f:
            encoder = encoder_cls.load_from_checkpoint(
                best_checkpoint / "model_weight.pkl",
                **json.load(f),
                autoencoder=base_autoencoder,
                latent_dim=encoder_train_config.latent_dim,
                criterion="balanced_tabular_recon",
                metrics=[],
                in_dim=data_module.x_dim,
                feature_mask=data_module.feature_mask,
                feature_categ_mask=data_module.feature_categ_mask,
                autoencoder_criterion="balanced_tabular_recon",
                autoencoder_metrics=[],
                classifier_criterion="cross_entropy",
                classifier_metrics=[],
                combined_metric_balance=1,
                out_dim=data_module.dataset_config.n_classes,
                map_location=device,
            )
    else:
        print(
            f"{data_module.dataset_config.identifier}, {encoder_tune_config.pretty_name} best model doesn't exist yet... "
        )
        return None, None
    return encoder, best_checkpoint
