import hydra
from omegaconf import DictConfig
from pathlib import Path
from dataclasses import dataclass

from tabdd.config.paths import ROOT_DIR


@dataclass
class EncoderTrainConfig:
    num_epochs: int
    num_samples: int
    latent_dim: int
    target_metric: str
    max_concurrent_trials: int
    criterion: str
    gpu_per_worker: int | float
    cpu_per_worker: int
    checkpoint_dir: Path
    classifier_target_metric: str
    classifier_criterion: str
    autoencoder_target_metric: str
    autoencoder_criterion: str
    combined_metric_balance: float
    train_target: list[str]

    def __post_init__(self):
        self.checkpoint_dir = ROOT_DIR / self.checkpoint_dir

    def update(self, other: "EncoderTrainConfig"):
        self.num_epochs = other.num_epochs
        self.num_samples = other.num_samples
        self.latent_dim = other.latent_dim
        self.target_metric = other.target_metric
        self.max_concurrent_trials = other.max_concurrent_trials
        self.criterion = other.criterion
        self.gpu_per_worker = other.gpu_per_worker
        self.cpu_per_worker = other.cpu_per_worker
        self.checkpoint_dir = other.checkpoint_dir

    @property
    def identifier(self):
        if (
            self.classifier_target_metric is None
            and self.autoencoder_target_metric is None
        ):
            # is not multihead
            return "-".join(
                [
                    self.target_metric,
                    self.criterion,
                    str(self.latent_dim),
                    str(self.num_samples),
                    str(self.num_epochs),
                ]
            )
        else:
            return "-".join(
                [
                    self.target_metric,
                    self.classifier_target_metric,
                    self.classifier_criterion,
                    self.autoencoder_target_metric,
                    self.autoencoder_criterion,
                    str(self.latent_dim),
                    str(self.num_samples),
                    str(self.num_epochs),
                ]
            )


@dataclass
class EncoderTuneConfig:
    encoder_name: str
    cls: callable
    tune_params: dict

    @property
    def identifier(self):
        return self.encoder_name.lower()

    @property
    def pretty_name(self):
        if "gnn" in self.encoder_name.lower():
            return "GNN"
        elif "mlp" in self.encoder_name.lower():
            return "MLP"
        elif "tf" in self.encoder_name.lower():
            return "TF"
        else:
            return "Unknown"


@dataclass
class MultiEncoderTuneConfig:
    base_encoder_config: EncoderTuneConfig
    tune_params: dict

    @property
    def identifier(self):
        return "multihead" + self.base_encoder_config.encoder_name.lower()

    @property
    def pretty_name(self):
        return self.base_encoder_config.pretty_name + "-MultiHead"


def load_encoder_train_config(config: DictConfig) -> EncoderTrainConfig:
    return hydra.utils.instantiate(config.encoder.train)


def load_encoder_tune_configs(
    config: DictConfig,
) -> list[EncoderTuneConfig | MultiEncoderTuneConfig]:

    encoder_configs = []

    if "encoder" not in config.keys() or any(
        k not in config.encoder.keys() for k in ["models", "train"]
    ):
        return encoder_configs

    encoder_configs += [
        hydra.utils.instantiate(
            enc_conf,
            _convert_="all",
        )
        for enc_conf in config.encoder.models.values()
    ]

    if "multihead" in config.encoder.train.train_target:
        multi_encoder_configs = [
            hydra.utils.instantiate(
                config.encoder.multihead,
                _partial_=True,
                _convert_="all",
            )(econf)
            for econf in encoder_configs
        ]

        if "base" not in config.encoder.train.train_target:
            encoder_configs = multi_encoder_configs
        else:
            encoder_configs += multi_encoder_configs

    return encoder_configs
