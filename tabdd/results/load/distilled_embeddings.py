from typing import Literal
from pathlib import Path
import pandas as pd
import numpy as np
import h5py

from tabdd.data import load_data_module
from tabdd.data.reduce_2d import reduce_2d
from tabdd.distill import load_distilled_data
from tabdd.config import DistillConfig, DataConfig


def load_distilled_embeddings(
    dataset_name: str,
    reduce_method: Literal["tsne", "umap", "pca"],
    distill_method: Literal[
        "random_sample", "kmeans", "closest_kmeans", "agglo", "closest_agglo", "kip"
    ],
    distill_space: Literal["encoded", "original"],
    output_space: Literal["encoded", "decoded"],
    convert_binary: bool,
    cluster_center: Literal["centroid", "closest"],
    distill_size: int,
    encoder_name: str = "original",
    latent_dim: int = 16,
    random_state: int = 0,
    cache_dir: str | Path | None = None,
):

    data_config = DataConfig(
        dataset_name=dataset_name,
    )

    distill_config = DistillConfig(
        data_config=data_config,
        distill_method=distill_method,
        distill_space=distill_space,
        output_space=output_space,
        convert_binary=convert_binary,
        cluster_center=cluster_center,
        distill_size=distill_size,
        encoder_name=encoder_name,
    )
    dm = load_data_module(data_config)

    if distill_method == "random_sample":
        data_mode = distill_method
    else:
        if distill_space == "original":
            data_mode = f"{distill_method}_{distill_space}"
        elif distill_space == "encoded":
            data_mode = f"{distill_method}_{distill_space}_{output_space}"
            if output_space == "decoded" and convert_binary:
                data_mode += "_binary"
        else:
            raise NotImplementedError(
                f"Distill method [{distill_method}], Distill space[{distill_space}] not found"
            )
        data_mode = data_mode + f"/{cluster_center}"

    if cache_dir is not None:
        cache_path = (
            Path(cache_dir) / data_config.identifier / data_mode / str(distill_size)
        ).resolve()

        if distill_method != "random_sample":
            cache_path = cache_path / encoder_name.lower() / str(latent_dim)

        cache_path.mkdir(exist_ok=True, parents=True)
        reduced_data_cache = (
            cache_path / f"distilled_{reduce_method}_{random_state:03d}.h5"
        )
    else:
        reduced_data_cache = None

    distilled_data = load_distilled_data(distill_config)
    X_train = distilled_data["train"].X
    y_train = distilled_data["train"].y

    if reduced_data_cache is not None and not reduced_data_cache.is_file():
        train_distilled_reduced = reduce_2d(
            X_train=X_train,
            reduce_method=reduce_method,
        )
        X_train_distilled_reduced = train_distilled_reduced["train"]

        if reduced_data_cache is not None:
            with open(reduced_data_cache, "wb") as f:
                with h5py.File(f, "w") as cache_file:
                    cache_file.create_dataset(
                        "X_train_reduced", data=X_train_distilled_reduced
                    )
    else:
        with open(reduced_data_cache, "rb") as f:
            with h5py.File(f, "r") as f:
                X_train_distilled_reduced = cache_file["X_train_reduced"][:]

    reduced_embs = pd.DataFrame(
        np.hstack([X_train_distilled_reduced, y_train[:, None]]),
        columns=["D1", "D2", "Label"],
    )

    reduced_embs["Encoder"] = encoder_name
    if distill_method == "random_sample":
        reduced_embs["Output Mode"] = "Random Sample"
    else:
        if convert_binary:
            reduced_embs["Output Mode"] = f"{output_space.capitalize()}-Binary"
        else:
            reduced_embs["Output Mode"] = f"{output_space.capitalize()}"

    return reduced_embs
