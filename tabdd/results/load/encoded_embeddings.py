from typing import Literal
from pathlib import Path
import pandas as pd
import numpy as np
import h5py

from tabdd.data import TabularDataModule
from tabdd.data.reduce_2d import reduce_2d
from tabdd.models.encoder import load_encoder
from tabdd.distill.random_sample import random_sample
from tabdd.config import DatasetConfig, DataModeConfig, EncoderTuneConfig


def load_encoded_embeddings(
    # dataset_name: str,
    # encoder_name: str,
    # scale_mode: str = 'standard',
    # parse_mode: str = 'onehot',
    # n_bins: int = 10,
    # bin_strat: str = 'uniform',
    dataset_config: DatasetConfig,
    data_mode_config: DataModeConfig,
    encoder_config: EncoderTuneConfig | None,
    reduce_method: Literal["tsne", "umap", "pca"],
    # use_original: bool = False,
    base_sample_size: int | float = 0.2,
    random_state: int = 0,
    cache_dir: str | Path | None = None,
):
    dm = TabularDataModule(
        dataset_config=dataset_config,
        data_mode_config=data_mode_config,
    )
    dm.prepare_data()
    dm.setup()

    encoder, _ = load_encoder(
        encoder_config=encoder_config,
        data_module=dm,
    )

    if cache_dir is not None:
        reduced_data_cache = (
            cache_dir
            / dm.identifier
            / encoder_config.identifier
            / str(encoder_config.latent_dim)
            / "encoded_embeddings"
            / f"encoded_{reduce_method}_{random_state:03d}.h5"
        ).resolve()
        reduced_data_cache.parent.mkdir(parents=True, exist_ok=True)
    else:
        reduced_data_cache = None

    sampled = {}
    for subset_name, subset in dm.subsets.items():
        sampled_idx, sampled_y = random_sample(
            X=np.arange(subset.y.shape[0])[:, None],
            y=subset.y.cpu().numpy(),
            N=base_sample_size,
            match_balance=True,
            random_state=random_state,
        )
        sampled[subset_name] = {"idx": sampled_idx.flatten(), "y": sampled_y}

    train_enc, val_enc, test_enc = dm.get_encoded_X(encoder)
    train_dec, val_dec, test_dec = dm.get_decoded_X(encoder)

    if reduced_data_cache is not None and not reduced_data_cache.is_file():
        ori_reduced = reduce_2d(
            X_train=dm.subsets["train"].X[sampled["train"]["idx"]],
            X_val=dm.subsets["val"].X[sampled["val"]["idx"]],
            X_test=dm.subsets["test"].X[sampled["test"]["idx"]],
            reduce_method=reduce_method,
        )

        enc_reduced = reduce_2d(
            X_train=train_enc[sampled["train"]["idx"]],
            X_val=val_enc[sampled["val"]["idx"]],
            X_test=test_enc[sampled["test"]["idx"]],
            reduce_method=reduce_method,
        )

        dec_reduced = reduce_2d(
            X_train=train_dec[sampled["train"]["idx"]],
            X_val=val_dec[sampled["val"]["idx"]],
            X_test=test_dec[sampled["test"]["idx"]],
            reduce_method=reduce_method,
        )

        if reduced_data_cache is not None:
            with open(reduced_data_cache, "wb") as f:
                with h5py.File(f, "w") as cache_file:
                    cache_file.create_dataset(
                        "ori_reduced_train", data=ori_reduced["train"]
                    )
                    cache_file.create_dataset(
                        "ori_reduced_val", data=ori_reduced["val"]
                    )
                    cache_file.create_dataset(
                        "ori_reduced_test", data=ori_reduced["test"]
                    )
                    cache_file.create_dataset(
                        "enc_reduced_train", data=enc_reduced["train"]
                    )
                    cache_file.create_dataset(
                        "enc_reduced_val", data=enc_reduced["val"]
                    )
                    cache_file.create_dataset(
                        "enc_reduced_test", data=enc_reduced["test"]
                    )
                    cache_file.create_dataset(
                        "dec_reduced_train", data=dec_reduced["train"]
                    )
                    cache_file.create_dataset(
                        "dec_reduced_val", data=dec_reduced["val"]
                    )
                    cache_file.create_dataset(
                        "dec_reduced_test", data=dec_reduced["test"]
                    )

    else:
        with open(reduced_data_cache, "rb") as f:
            with h5py.File(f, "r") as cache_file:
                ori_reduced = {
                    "train": cache_file["ori_reduced_train"][:],
                    "val": cache_file["ori_reduced_val"][:],
                    "test": cache_file["ori_reduced_test"][:],
                }
                enc_reduced = {
                    "train": cache_file["enc_reduced_train"][:],
                    "val": cache_file["enc_reduced_val"][:],
                    "test": cache_file["enc_reduced_test"][:],
                }
                dec_reduced = {
                    "train": cache_file["dec_reduced_train"][:],
                    "val": cache_file["dec_reduced_val"][:],
                    "test": cache_file["dec_reduced_test"][:],
                }
    result = pd.concat(
        [
            pd.DataFrame(
                np.hstack(
                    [ori_reduced[subset_name], sampled[subset_name]["y"][:, None]]
                ),
                columns=["D1", "D2", "Label"],
            ).assign(**{"Subset": subset_name.capitalize(), "Space": "Original"})
            for subset_name in dm.subsets
        ]
        + [
            pd.DataFrame(
                np.hstack(
                    [enc_reduced[subset_name], sampled[subset_name]["y"][:, None]]
                ),
                columns=["D1", "D2", "Label"],
            ).assign(**{"Subset": subset_name.capitalize(), "Space": "Encoded"})
            for subset_name in dm.subsets
        ]
        + [
            pd.DataFrame(
                np.hstack(
                    [dec_reduced[subset_name], sampled[subset_name]["y"][:, None]]
                ),
                columns=["D1", "D2", "Label"],
            ).assign(**{"Subset": subset_name.capitalize(), "Space": "Decoded"})
            for subset_name in dm.subsets
        ]
    )
    return result
