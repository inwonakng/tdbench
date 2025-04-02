from omegaconf import DictConfig
import pandas as pd
import numpy as np
from rich.progress import Progress

from tabdd.data import TabularDataModule

from tabdd.config.paths import DATA_DIR
from tabdd.config import (
    load_dataset_configs,
    load_data_mode_config,
    DatasetConfig,
    DataModeConfig,
)


def load_dataset_stats(
    # data_config: DataConfig,
    dataset_config: DatasetConfig,
    data_mode_config: DataModeConfig,
    progress: Progress | None = None,
):
    dm = TabularDataModule(
        dataset_config=dataset_config,
        data_mode_config=data_mode_config,
    )

    dm.prepare_data()
    dm.setup()
    labels, labels_count = np.unique(dm.y, return_counts=True)

    stats = {
        "Dataset": dataset_config.dataset_name,
        "Rows": dm.num_rows,
        "Original Features": dm.feature_categ_mask.shape[0],
        "Categorical Features": dm.feature_categ_mask.sum(),
        "Continuous Features": (~dm.feature_categ_mask).sum(),
        "Features after Transform": dm.feature_mask.shape[0],
        "Labels": labels.shape[0],
        "Label Ratio": tuple(labels_count),
        "Train Size": len(dm.subsets["train"]),
        "Val Size": len(dm.subsets["val"]),
        "Test Size": len(dm.subsets["test"]),
        "Minority Label Ratio": min(labels_count) / sum(labels_count),
    }

    return pd.DataFrame([stats])


def load_all_dataset_stats(config: DictConfig):
    data_mode_config = load_data_mode_config(config)
    dataset_configs = load_dataset_configs(config)

    ds_stats = pd.concat(
        [
            load_dataset_stats(
                dataset_config=dataset_config, data_mode_config=data_mode_config
            )
            for dataset_config in dataset_configs
            # for dataset in DATASETS
        ]
    ).reset_index(drop=True)
    return ds_stats
