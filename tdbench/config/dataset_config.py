import hydra
from omegaconf import DictConfig
from dataclasses import dataclass


@dataclass
class DatasetConfig:
    download_url: str
    label: str
    n_classes: int
    source_type: str
    dataset_name: str

    @property
    def identifier(self):
        return f"{self.dataset_name}"


def load_dataset_configs(config: DictConfig) -> list[DatasetConfig]:
    return [
        hydra.utils.instantiate(dataset) for dataset in config.data.datasets.values()
    ]
