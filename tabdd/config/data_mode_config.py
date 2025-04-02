import hydra
from typing import Literal
from omegaconf import DictConfig
from dataclasses import dataclass
from pathlib import Path

from tabdd.config.paths import DATA_DIR


@dataclass
class DataModeConfig:
    parse_mode: Literal["onehot", "mixed", "ple"] = "onehot"
    scale_mode: str = "standard"
    bin_strat: str = "uniform"
    n_bins: int = 10
    data_dir: Path = Path(DATA_DIR)
    batch_size: int = 1024
    val_ratio: float = 0.15
    test_ratio: float = 0.15

    @property
    def identifier(self):
        return f"{self.parse_mode}/{self.scale_mode}/{self.bin_strat}-{self.n_bins}"


def load_data_mode_config(config: DictConfig) -> DataModeConfig:
    return hydra.utils.instantiate(config.data.mode, _convert_="all")
