import hydra
from omegaconf import DictConfig
from dataclasses import dataclass
from pathlib import Path

from tabdd.config.paths import (
    RAY_RESULT_DIR,
    DATA_REPO_DIR,
)
from tabdd.models.classifier import get_classifier


@dataclass
class ClassifierTuneConfig:
    classifier_name: str
    default_params: dict
    tune_params: dict
    use_sample_weight: bool
    use_n_jobs: bool

    @property
    def identifier(self):
        return self.classifier_name

    def set(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

    def instantiate(
        self,
        params: dict,
    ):
        return get_classifier(
            classifier_name=self.classifier_name,
            default_params=self.default_params,
            params=params,
        )


@dataclass
class ClassifierTrainConfig:
    num_samples: int
    metric_name: str
    optimizer_name: str
    n_folds: int
    results_dir: Path | str
    tune_hyperopt: bool
    rerun_tune: bool
    cpu_per_worker: int
    max_concurrent_trials: int
    storage_dir: Path = RAY_RESULT_DIR

    def __post_init__(self):
        self.results_dir = DATA_REPO_DIR / self.results_dir

    @property
    def identifier(self):
        return "metric={}-opt={}-n_samples={}-n_folds={}-tune_hyperopt={}".format(
            self.metric_name,
            self.optimizer_name,
            self.num_samples,
            self.n_folds,
            self.tune_hyperopt,
        )


def load_classifier_tune_configs(config: DictConfig) -> list[ClassifierTuneConfig]:
    return [
        hydra.utils.instantiate(clf_conf, _convert_="all")
        for clf_conf in config.classifier.models.values()
    ]


def load_classifier_train_config(config: DictConfig) -> ClassifierTrainConfig:
    return hydra.utils.instantiate(config.classifier.train, _convert_="all")
