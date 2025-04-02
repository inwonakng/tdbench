import json
import pickle

import h5py
import numpy as np
import pytorch_lightning as pl
import torch
from rich.progress import Progress
from rtdl_num_embeddings import (
    PiecewiseLinearEncoding,
)
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from tabdd.config import DataModeConfig, DatasetConfig
from tabdd.models.encoder import BaseEncoder

from .load import load_openml_dataset
from .preprocessing import (
    KBinsDiscretizer,
    MaxAbsScaler,
    StandardScaler,
    build_one_hot_encoder,
    build_scaler,
)
from .tabular_dataset import TabularDataset

SUBSET_ENUM = {
    "train": 0,
    "val": 1,
    "test": 2,
}


class TabularDataModule(pl.LightningDataModule):
    dataset_config: DatasetConfig
    data_mode_config: DataModeConfig

    # stuff that gets populated from self.perpare()
    feature_mask: np.ndarray = None
    feature_categ_mask: np.ndarray = None
    columns: list[str] = None
    x_dim: int = None

    # stuff that gets populated from self.setup()
    num_features: int = None
    num_rows: int = None
    y: np.ndarray = None
    X: np.ndarray = None
    feature_idx: np.ndarray = None
    split_mask: np.ndarray = None
    subsets: dict[str, TabularDataset] = None
    loaders: dict[str, DataLoader] = None

    def __init__(
        self,
        dataset_config: DatasetConfig,
        data_mode_config: DataModeConfig,
        progress: Progress | None = None,
        **kwargs,
    ) -> None:
        super().__init__()
        if (
            data_mode_config.parse_mode not in ["mixed", "onehot", "ple"]
            or data_mode_config.scale_mode not in ["maxabs", "standard"]
            or data_mode_config.bin_strat not in ["uniform", "quantile", "kmeans"]
        ):
            raise NotImplementedError

        self.dataset_config = dataset_config
        self.data_mode_config = data_mode_config

    @property
    def scaler_save_dir(self):
        return self.raw_data_dir / f"{self.data_mode_config.scale_mode}.pkl"

    @property
    def onehot_encoder_save_dir(self):
        return (
            self.raw_data_dir
            / f"{self.data_mode_config.bin_strat}_{self.data_mode_config.n_bins}.pkl"
        )

    @property
    def data_dir(self):
        return self.data_mode_config.data_dir

    @property
    def identifier(self):
        return f"{self.dataset_config.dataset_name}/{self.data_mode_config.identifier}"

    @property
    def raw_data_dir(self):
        return self.data_dir / self.dataset_config.identifier

    @property
    def processed_cache_dir(self):
        return self.data_dir / self.identifier

    @property
    def feature_categ_mask_path(self):
        return (self.raw_data_dir / "feature_categ_mask.json").resolve()

    @property
    def columns_path(self):
        return (self.raw_data_dir / "columns.json").resolve()

    @property
    def feature_mask_path(self):
        return (self.processed_cache_dir / "feature_mask.json").resolve()

    @property
    def cache_dir(self):
        return self.processed_cache_dir / "processed.h5"

    def load_data(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Loads the raw data and parses it into tensors to consume in following steps.
            Returns a tuple of 4 variables: `X`, `y`, `feature_categ_mask` and `split_mask`.
            `feature_categ_mask` is a mask to specify whether X's feature in the index is categorical or not.

        Returns:
            tuple[torch.FloatTensor, torch.LongTensor, torch.LongTensor, torch.LongTensor]: `X (num_row, num_col)`, `y (num_row)`, `feature_categ_mask (num_col)`, `split_mask (num_row)`
        """
        self.raw_data_dir.mkdir(exist_ok=True, parents=True)
        data_save_dir = self.raw_data_dir / "dataset.h5"
        if not data_save_dir.is_file():
            if self.dataset_config.source_type == "openml":
                X, y = load_openml_dataset(
                    dataset_dir=self.raw_data_dir,
                    download_url=self.dataset_config.download_url,
                    label=self.dataset_config.label,
                )
            # elif self.dataset_config.source_type=='kaggle':
            #     X, y = load_kaggle_dataset(
            #         dataset_dir,
            #         self.dataset_config.identifier,
            #         self.file_name,
            #         self.dataset_config.label
            #     )
            else:
                raise NotImplementedError(
                    f"Data source {self.dataset_config.source_type} is not implemented"
                )
            # assume all int types are categorical
            feature_categ_mask = (
                X.dtypes.isin([np.int8, np.int16, np.int32, np.int64])
            ).values.astype(bool)
            split_mask = self.get_split_mask(
                y, self.data_mode_config.val_ratio, self.data_mode_config.test_ratio
            )
            columns = X.columns.tolist()
            X = X.values
            y = y.values
            np.savez_compressed(data_save_dir, X=X, y=y, split_mask=split_mask)
            json.dump(
                feature_categ_mask.tolist(),
                open(self.feature_categ_mask_path, "w"),
            )
            with open(data_save_dir, "wb") as f:
                with h5py.File(f, "w") as cache_file:
                    cache_file.create_dataset("X", data=X)
                    cache_file.create_dataset("y", data=y)
                    cache_file.create_dataset("split_mask", data=split_mask)
            with open(self.feature_categ_mask_path, "w") as f:
                json.dump(feature_categ_mask.tolist(), f)
            with open(self.columns_path, "w") as f:
                json.dump(columns, f)
        else:
            with open(data_save_dir, "rb") as f:
                with h5py.File(f, "r") as cache_file:
                    X = cache_file["X"][:]
                    y = cache_file["y"][:]
                    split_mask = cache_file["split_mask"][:]
            with open(self.feature_categ_mask_path) as f:
                feature_categ_mask = np.array(json.load(f)).astype(bool)
            with open(self.columns_path) as f:
                columns = json.load(f)

        return X, y, feature_categ_mask, split_mask, columns

    @staticmethod
    def get_split_mask(
        y: np.ndarray,
        val_ratio: float,
        test_ratio: float,
    ) -> np.ndarray:
        """Creates the split mask to use to build split subsets of the dataset.

        Args:
            y (np.ndarray): The label column to use in stratifying (matching the balance of labels in each split).

        Returns:
            torch.LongTensor: A 1-d tensor composed of 0, 1 and 2 for train, validation and test.
        """
        train_idxs, val_test_idxs = train_test_split(
            np.arange(len(y)),
            stratify=y,
            test_size=val_ratio + test_ratio,
            random_state=0,
            shuffle=True,
        )
        val_idxs, test_idxs = train_test_split(
            val_test_idxs,
            stratify=y[val_test_idxs],
            test_size=val_ratio / (val_ratio + test_ratio),
            random_state=0,
            shuffle=True,
        )
        split_mask = np.zeros(len(y)).astype(int)
        split_mask[val_idxs] = SUBSET_ENUM["val"]
        split_mask[test_idxs] = SUBSET_ENUM["test"]
        return split_mask

    def load_onehot_encoder(self):
        if self.data_mode_config.parse_mode not in ["onehot", "ple"]:
            raise Exception(
                f"Onehot encoder is not needed for mode [{self.data_mode_config.parse_mode}]"
            )
        if not (~self.feature_categ_mask).any():
            raise Exception("Dataset doesn't have any continuous features!")
        if not self.onehot_encoder_save_dir.is_file():
            raise Exception(
                f"Cannot find pre-fitted onehot encoder for [{self.dataset_config.dataset_name}]"
            )
        with open(self.onehot_encoder_save_dir, "rb") as f:
            onehot_encoder = pickle.load(f)
        return onehot_encoder

    def load_scaler(self):
        if not (~self.feature_categ_mask).any() or (not self.scaler_save_dir.is_file()):
            scaler = None
        else:
            with open(self.scaler_save_dir, "rb") as f:
                scaler = pickle.load(f)
        return scaler

    def undo_onehot(self, X: np.ndarray):
        onehot_encoder = self.load_onehot_encoder()
        if onehot_encoder is None:
            raise ValueError(f"No onehot encoder found for {self.identifier}")
        return X

    def apply_onehot(self, X: np.ndarray):
        onehot_encoder = self.load_onehot_encoder()
        if onehot_encoder is None:
            raise ValueError(f"No onehot encoder found for {self.identifier}")
        return onehot_encoder.transform(X).toarray()

    def build_processors(
        self,
        X: torch.FloatTensor | np.ndarray,
        feature_categ_mask: np.ndarray,
        split_mask: torch.LongTensor,
    ) -> tuple[StandardScaler | MaxAbsScaler, KBinsDiscretizer]:
        """Builds the scaler and one-hot encoder which is used by the TabularDataset instance of each subset.

        Args:
            X (torch.FloatTensor): Unmodified `X (num_row, num_col)` tensor.
            feature_categ_mask (torch.LongTensor): A 1-d tensor of length `num_col` to specify whether X's feature in the index is categorical or not.
            split_mask (torch.LongTensor): A 1-d tensor of length `num_row` composed of 0, 1 and 2 for train, validation and test.

        Returns:
            tuple[StandardScaler | MaxAbsScaler, KBinsDiscretizer]: _description_
        """

        if (~feature_categ_mask).any():
            if not self.scaler_save_dir.is_file():
                scaler = build_scaler(self.data_mode_config.scale_mode)
                scaler.fit(
                    X[split_mask == SUBSET_ENUM["train"]][:, ~feature_categ_mask]
                )
                with open(self.scaler_save_dir, "wb") as f:
                    pickle.dump(scaler, f)
            else:
                with open(self.scaler_save_dir, "rb") as f:
                    scaler = pickle.load(f)
        else:
            scaler = None

        if (
            self.data_mode_config.parse_mode in ["onehot", "ple"]
            and (~feature_categ_mask).any()
        ):
            if not self.onehot_encoder_save_dir.is_file():
                onehot_encoder = build_one_hot_encoder(
                    self.data_mode_config.bin_strat,
                    self.data_mode_config.n_bins,
                )
                onehot_encoder.fit(
                    scaler.transform(
                        X[split_mask == SUBSET_ENUM["train"]][:, ~feature_categ_mask]
                    )
                )
                with open(self.onehot_encoder_save_dir, "wb") as f:
                    pickle.dump(onehot_encoder, f)
            else:
                with open(self.onehot_encoder_save_dir, "rb") as f:
                    onehot_encoder = pickle.load(f)
        else:
            onehot_encoder = None

        return scaler, onehot_encoder

    def build_feature_mask(
        self,
        X: np.ndarray,
        feature_categ_mask: np.ndarray,
        onehot_encoder: KBinsDiscretizer,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Builds the 1-d array to mark while indices belong to a feature.
        feature_mask is a array that looks like [0, 0, 0, 1, 1, 1, 1, ...] to denote the binaries features that belong together in the original column.

        Args:
            X (np.ndarray): Unmodified `X (num_row, num_col)` array.
            feature_categ_mask (np.ndarray): A 1-d array of length `num_col` to specify whether X's feature in the index is categorical or not.

        Returns:
            tuple[np.ndarray, np.ndarray, np.ndarray]: `feature_mask (num_bin)`, `binary_categ_mask (num_bin)`, `feature_offsets (num_col-1)`
        """
        feature_mask, binary_categ_mask, feature_offsets = [], [], [0]
        cont_bin_sizes = (
            [b.shape[0] - 1 for b in onehot_encoder.bin_edges_]
            if onehot_encoder
            else []
        )
        for i, is_categ in enumerate(feature_categ_mask):
            one_feature_mask = []
            if is_categ:
                one_feature_mask = [i] * len(np.unique(X[:, i]))
            elif self.data_mode_config.parse_mode in ["onehot", "ple"]:
                one_feature_mask = [i] * cont_bin_sizes[(~feature_categ_mask[:i]).sum()]
            elif self.data_mode_config.parse_mode == "mixed":
                one_feature_mask = [i]
            feature_mask += one_feature_mask
            binary_categ_mask += [is_categ] * len(one_feature_mask)
            feature_offsets += [len(feature_mask)]
        feature_mask = np.array(feature_mask).astype(int)
        binary_categ_mask = np.array(binary_categ_mask).astype(bool)
        feature_offsets = np.array(feature_offsets[:-1]).astype(int)
        json.dump(feature_mask.tolist(), open(self.feature_mask_path, "w"))
        return feature_mask, binary_categ_mask, feature_offsets

    def prepare_data(self) -> None:
        """Calls the helper functions to prepare the subsets. Each subset is a TabularDataset instance."""

        self.processed_cache_dir.mkdir(exist_ok=True, parents=True)
        if not self.cache_dir.is_file():
            self.cache_dir.parent.mkdir(exist_ok=True, parents=True)
            X, y, feature_categ_mask, split_mask, columns = self.load_data()
            scaler, onehot_encoder = self.build_processors(
                X, feature_categ_mask, split_mask
            )
            feature_mask, binary_categ_mask, feature_offsets = self.build_feature_mask(
                X, feature_categ_mask, onehot_encoder
            )

            # do the preprocessing here
            X_encoded = np.zeros((X.shape[0], len(feature_mask)))
            self.x_dim = len(feature_mask)
            self.feature_categ_mask = feature_categ_mask
            self.feature_mask = feature_mask
            self.columns = columns

            scaled_cont = (
                scaler.transform(X[:, ~feature_categ_mask])
                if scaler
                else X[:, ~feature_categ_mask]
            )

            # Turning the categorical features to onehot first
            # set the categ feature indices to 1
            np.put_along_axis(
                X_encoded,
                ((feature_offsets + X)[:, feature_categ_mask]).astype(int),
                values=1,
                axis=1,
            )

            if onehot_encoder:
                if self.data_mode_config.parse_mode == "onehot":
                    X_encoded = X_encoded.astype(int)
                    if (~feature_categ_mask).sum() > 0:
                        encoded_cont = onehot_encoder.transform(scaled_cont).toarray()
                        X_encoded[:, ~binary_categ_mask] = encoded_cont
                    feature_idx = np.vstack([row.nonzero()[0] for row in X_encoded])
                elif self.data_mode_config.parse_mode == "ple":
                    ple_enc = PiecewiseLinearEncoding(
                        bins=[
                            torch.tensor(e, dtype=torch.float32)
                            for e in onehot_encoder.bin_edges_
                        ]
                    )
                    X_encoded[:, ~binary_categ_mask] = ple_enc(
                        torch.tensor(scaled_cont, dtype=torch.float32)
                    ).numpy()
                    all_idxs = np.arange(X_encoded.shape[1])
                    feature_idx = np.vstack(
                        [all_idxs[(row != 0) | ~binary_categ_mask] for row in X_encoded]
                    )
                else:
                    raise Exception("Unknown encoding strategy")
            else:
                X_encoded = X_encoded.astype(float)
                X_encoded[:, ~binary_categ_mask] = scaled_cont
                feature_idx = np.vstack(
                    [
                        np.arange(len(row))[(row != 0) | ~binary_categ_mask]
                        for row in X_encoded
                    ]
                )
            # there can be a case where there is no one hot encoder (every feature is categorical) but we still want the data in one hot form. Turn to in in that case
            if self.data_mode_config.parse_mode == "onehot":
                X_encoded = X_encoded.astype(int)
            if self.data_mode_config.parse_mode == "ple":
                X_encoded = X_encoded.astype(float)
            with open(self.cache_dir, "wb") as f:
                with h5py.File(f, "w") as cache_file:
                    cache_file.create_dataset("X_encoded", data=X_encoded)
                    cache_file.create_dataset("y", data=y)
                    cache_file.create_dataset("feature_idx", data=feature_idx)
                    cache_file.create_dataset("split_mask", data=split_mask)
        else:
            with open(self.feature_mask_path, "r") as f:
                self.feature_mask = np.array(json.load(f)).astype(int)
            with open(self.feature_categ_mask_path) as f:
                self.feature_categ_mask = np.array(json.load(f)).astype(bool)
            with open(self.columns_path) as f:
                self.columns = json.load(f)
            self.x_dim = len(self.feature_mask)

    def setup(self, stage: str = ""):
        with open(self.cache_dir, "rb") as f:
            with h5py.File(f, "r") as cache_file:
                X_encoded = cache_file["X_encoded"][:]
                y = cache_file["y"][:]
                feature_idx = cache_file["feature_idx"][:]
                split_mask = cache_file["split_mask"][:]

        self.num_features = len(self.feature_categ_mask)
        self.num_rows = X_encoded.shape[0]
        self.y = y
        self.X = X_encoded
        self.feature_idx = feature_idx
        self.split_mask = split_mask

        X_encoded = torch.from_numpy(X_encoded)
        y = torch.from_numpy(y).long()
        feature_idx = torch.from_numpy(feature_idx).long()

        # return
        self.subsets = {
            subset: TabularDataset(
                X=X_encoded[split_mask == SUBSET_ENUM[subset]],
                y=y[split_mask == SUBSET_ENUM[subset]],
                feature_mask=self.feature_mask,
                feature_categ_mask=self.feature_categ_mask,
                feature_idx=feature_idx[split_mask == SUBSET_ENUM[subset]],
            )
            for subset in ["train", "val", "test"]
        }

        self.loaders = {
            subset: DataLoader(
                self.subsets[subset],
                batch_size=(
                    self.data_mode_config.batch_size
                    if self.data_mode_config.batch_size > 0
                    else len(self.subsets["train"])
                ),
                shuffle=(subset == "train"),
                num_workers=0,
            )
            for subset in ["train", "val", "test"]
        }

    def load(self) -> None:
        self.prepare_data()
        self.setup()

    def train_dataloader(self) -> DataLoader:
        return self.loaders["train"]

    def val_dataloader(self) -> DataLoader:
        return self.loaders["val"]

    def test_dataloader(self) -> DataLoader:
        return self.loaders["test"]

    def get_encoded_X(
        self,
        encoder: BaseEncoder,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        encoded_cache_dir = (
            self.processed_cache_dir
            / encoder.name.lower()
            / str(encoder.latent_dim)
            / "encoded.h5"
        ).resolve()
        encoded_cache_dir.parent.mkdir(exist_ok=True, parents=True)
        if not encoded_cache_dir.is_file():
            # NOTE: sometimes this goes over the memory limit of 1 A100 (80GB), need to batch
            # We will use manual batching since the loaders add shuffle.
            batch_size = 1024

            # Train data
            train_idxs = torch.arange(len(self.subsets["train"].X))
            train_encoded = [
                encoder.encode(
                    self.subsets["train"].X[batch],
                    feature_idx=self.subsets["train"].feature_idx[batch],
                    feature_mask=self.feature_mask,
                    feature_categ_mask=self.feature_categ_mask,
                )
                .detach()
                .cpu()
                for batch in train_idxs.split(batch_size)
            ]
            train_encoded = torch.vstack(train_encoded).numpy()

            val_idxs = torch.arange(len(self.subsets["val"].X))
            val_encoded = [
                encoder.encode(
                    self.subsets["val"].X[batch],
                    feature_idx=self.subsets["val"].feature_idx[batch],
                    feature_mask=self.feature_mask,
                    feature_categ_mask=self.feature_categ_mask,
                )
                .detach()
                .cpu()
                for batch in val_idxs.split(batch_size)
            ]
            val_encoded = torch.vstack(val_encoded).numpy()

            test_idxs = torch.arange(len(self.subsets["test"].X))
            test_encoded = [
                encoder.encode(
                    self.subsets["test"].X[batch],
                    feature_idx=self.subsets["test"].feature_idx[batch],
                    feature_mask=self.feature_mask,
                    feature_categ_mask=self.feature_categ_mask,
                )
                .detach()
                .cpu()
                for batch in test_idxs.split(batch_size)
            ]
            test_encoded = torch.vstack(test_encoded).numpy()

            with open(encoded_cache_dir, "wb") as f:
                with h5py.File(f, "w") as cache_file:
                    cache_file.create_dataset("train_encoded", data=train_encoded)
                    cache_file.create_dataset("val_encoded", data=val_encoded)
                    cache_file.create_dataset("test_encoded", data=test_encoded)
        else:
            with open(encoded_cache_dir, "rb") as f:
                with h5py.File(f, "r") as cache_file:
                    train_encoded = cache_file["train_encoded"][:]
                    val_encoded = cache_file["val_encoded"][:]
                    test_encoded = cache_file["test_encoded"][:]

        return train_encoded, val_encoded, test_encoded

    def get_decoded_X(
        self,
        encoder: BaseEncoder,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        decoded_cache_dir = (
            self.processed_cache_dir
            / encoder.name.lower()
            / str(encoder.latent_dim)
            / "decoded.h5"
        ).resolve()
        if not decoded_cache_dir.is_file():
            train_encoded, val_encoded, test_encoded = self.get_encoded_X(encoder)
            train_decoded = (
                encoder.convert_binary(encoder.decode(torch.tensor(train_encoded)))
                .detach()
                .cpu()
                .numpy()
            )
            val_decoded = (
                encoder.convert_binary(encoder.decode(torch.tensor(val_encoded)))
                .detach()
                .cpu()
                .numpy()
            )
            test_decoded = (
                encoder.convert_binary(encoder.decode(torch.tensor(test_encoded)))
                .detach()
                .cpu()
                .numpy()
            )
            with open(decoded_cache_dir, "wb") as f:
                with h5py.File(f, "w") as cache_file:
                    cache_file.create_dataset("train_decoded", data=train_decoded)
                    cache_file.create_dataset("val_decoded", data=val_decoded)
                    cache_file.create_dataset("test_decoded", data=test_decoded)
        else:
            with open(decoded_cache_dir, "rb") as f:
                with h5py.File(f, "r") as cache_file:
                    train_decoded = cache_file["train_decoded"][:]
                    val_decoded = cache_file["val_decoded"][:]
                    test_decoded = cache_file["test_decoded"][:]

        return train_decoded, val_decoded, test_decoded
