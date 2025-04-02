import hydra
from typing import Literal
from omegaconf import DictConfig, OmegaConf
from dataclasses import dataclass, field
from pathlib import Path

from .paths import RUN_CONFIG_DIR
from .data_mode_config import DataModeConfig


def pretty_distill_method_name(distill_method: str) -> str:
    if distill_method == "original":
        return "Original"
    elif distill_method == "decoded":
        return "Decoded"
    elif distill_method == "encoded":
        return "Encoded"
    elif distill_method == "random_sample":
        return "Random Sample"
    elif distill_method == "agglo":
        return "Agglo"
    elif distill_method == "kmeans":
        return "KMeans"
    elif distill_method == "kip":
        return "KIP"
    elif distill_method == "gm":
        return "GM"
    elif distill_method == "forgetting":
        return "Forgetting"
    elif distill_method == "grand":
        return "GraNd"
    elif distill_method == "glister":
        return "Glister"
    elif distill_method == "graph_cut":
        return "GraphCut"
    elif distill_method == "mtt":
        return "MTT"
    elif distill_method == "datm":
        return "DATM"
    else:
        raise NotImplementedError(f"Distill method [{distill_method}] not found")


@dataclass
class DistillConfig:
    distill_method_name: str
    is_random: bool
    is_cluster: bool
    is_baseline: bool
    can_use_encoder: bool
    can_distill: bool
    args: dict = field(default_factory=dict)
    random_iters: int = 0
    distill_sizes: list[int] = field(
        default_factory=lambda _: [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    )
    output_spaces: tuple[str] = ("encoded", "decoded", "original")
    distill_spaces: tuple[str] = ("encoded", "original")
    post_data_mode_name: str = None
    post_data_mode_config: DataModeConfig = None

    def __post_init__(self):
        if self.post_data_mode_name is None:
            pass
        else:
            # check if file first
            f = Path(RUN_CONFIG_DIR) / f"data/mode/{self.post_data_mode_name}.yaml"
            if not f.is_file():
                raise Exception(f"{self.post_data_mode_name} is not a file..")
            self.post_data_mode_config = hydra.utils.instantiate(OmegaConf.load(f))

    @property
    def identifier(self):
        return self.distill_method_name.lower()

    @property
    def pretty_name(self):
        return pretty_distill_method_name(self.distill_method_name)


def load_distill_configs(config: DictConfig) -> list[DistillConfig]:
    return [
        hydra.utils.instantiate(
            dm_config,
            **config.distill.common,
            _convert_="all",
        )
        for dm_config in config.distill.methods.values()
    ]
