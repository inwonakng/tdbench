import torch
from sklearn.preprocessing import StandardScaler
import h5py
import time
import numpy as np

from tabdd.models.encoder import load_encoder
from tabdd.data import TabularDataset, TabularDataModule
from tabdd.config import (
    PipelineConfig,
)
from tabdd.utils import setup_logger

from .random_sample import random_sample
from .kmeans import kmeans
from .kip import kip
from .agglomerative import agglomerative
from .gradient_matching import gradient_matching
from .forgetting import forgetting
from .grand import grand
from .glister import glister
from .graph_cut import graph_cut
from .trajectory_matching import trajectory_matching
from .datm import datm

logger = setup_logger()


def load_distilled_data(
    pipeline_config: PipelineConfig,
    use_cache: bool = True,
) -> dict[str, TabularDataset]:
    data_module = TabularDataModule(
        dataset_config=pipeline_config.dataset_config,
        data_mode_config=pipeline_config.data_mode_config,
    )
    data_module.load()
    time_to_distill = 0
    if (
        pipeline_config.distill_space == "encoded"
        or pipeline_config.distill_method_name in ["encoded", "decoded"]
    ):
        encoder, _ = load_encoder(
            encoder_tune_config=pipeline_config.encoder_tune_config,
            encoder_train_config=pipeline_config.encoder_train_config,
            data_module=data_module,
        )
    else:
        encoder = None

    X_train, y_train = data_module.subsets["train"].X, data_module.subsets["train"].y
    X_val, y_val = data_module.subsets["val"].X, data_module.subsets["val"].y
    X_test, y_test = data_module.subsets["test"].X, data_module.subsets["test"].y
    feature_mask = data_module.feature_mask
    feature_categ_mask = data_module.feature_categ_mask
    original_X_train, original_y_train = X_train, y_train

    logger.info("-" * 40)
    logger.info(f"Distill Method: {pipeline_config.distill_method_name}")

    if pipeline_config.distill_method_name == "original":
        pass

    elif pipeline_config.distill_method_name == "encoded":
        X_train, X_val, X_test = data_module.get_encoded_X(encoder)
        feature_mask = np.arange(X_train.shape[1])
        feature_categ_mask = np.array([False for _ in feature_mask])
        original_X_train = X_train

    elif pipeline_config.distill_method_name == "decoded":
        X_train, X_val, X_test = data_module.get_decoded_X(encoder)

    elif pipeline_config.distill_method_name == "random_sample":
        if (
            not pipeline_config.distilled_data_dir.is_file()
            or not pipeline_config.distill_time_dir.is_file()
        ):
            start = time.time()
            X_train, y_train = random_sample(
                X=data_module.subsets["train"].X.cpu().numpy(),
                y=data_module.subsets["train"].y.cpu().numpy(),
                N=pipeline_config.distill_size,
                random_state=pipeline_config.random_seed,
            )
            time_to_distill = time.time() - start
            pipeline_config.distilled_data_dir.parent.mkdir(parents=True, exist_ok=True)
            with open(pipeline_config.distilled_data_dir, "wb") as f:
                with h5py.File(f, "w") as cache_file:
                    cache_file.create_dataset("X_train", data=X_train)
                    cache_file.create_dataset("y_train", data=y_train)
            pipeline_config.save_distill_time(time_to_distill)
        else:
            with open(pipeline_config.distilled_data_dir, "rb") as f:
                with h5py.File(f, "r") as cache_file:
                    X_train = cache_file["X_train"][:]
                    y_train = cache_file["y_train"][:]

    else:
        X_train, y_train = X_train.cpu().numpy(), y_train.cpu().numpy()
        X_val, y_val = X_val.cpu().numpy(), y_val.cpu().numpy()
        X_test, y_test = X_test.cpu().numpy(), y_test.cpu().numpy()
        distilled_idxs = None
        original_X_train, original_y_train = X_train.copy(), y_train.copy()
        if pipeline_config.distill_space == "original":
            logger.info(f"Distill Space: {pipeline_config.distill_space}")
            X_train_to_distill = X_train
            scaler = None
        elif pipeline_config.distill_space == "encoded":
            logger.info(f"Distill Space: {pipeline_config.distill_space}")
            (
                X_train_to_distill,
                X_val_encoded,
                X_test_encoded,
            ) = data_module.get_encoded_X(encoder)
            scaler = StandardScaler()
            X_train_to_distill = scaler.fit_transform(X_train_to_distill)
            if pipeline_config.output_space == "encoded":
                feature_mask = np.arange(X_train_to_distill.shape[1])
                feature_categ_mask = np.array([False for _ in feature_mask])
        else:
            raise NotImplementedError(
                f"distill space [{pipeline_config.distill_space}] is not implmented"
            )

        if (
            not pipeline_config.distilled_data_dir.is_file()
            or not pipeline_config.distill_time_dir.is_file()
            or not use_cache
        ):
            logger.info("Generating from scratch")
            if pipeline_config.distill_method_name == "kmeans":
                start = time.time()
                X_train, y_train, distilled_idxs = kmeans(
                    X=X_train_to_distill,
                    y=y_train,
                    N=pipeline_config.distill_size,
                    random_state=pipeline_config.random_seed,
                    get_closest=(pipeline_config.cluster_center == "closest"),
                    **pipeline_config.distill_config.args,
                )
                time_to_distill = time.time() - start

            elif pipeline_config.distill_method_name == "agglo":
                start = time.time()
                X_train, y_train, distilled_idxs = agglomerative(
                    X=X_train_to_distill,
                    y=y_train,
                    N=pipeline_config.distill_size,
                    random_state=pipeline_config.random_seed,
                    get_closest=(pipeline_config.cluster_center == "closest"),
                    **pipeline_config.distill_config.args,
                )
                time_to_distill = time.time() - start

            elif pipeline_config.distill_method_name == "kip":
                start = time.time()
                X_train, y_train = kip(
                    X=X_train_to_distill,
                    y=y_train,
                    N=pipeline_config.distill_size,
                    random_state=pipeline_config.random_seed,
                    **pipeline_config.distill_config.args,
                )
                time_to_distill = time.time() - start

            elif pipeline_config.distill_method_name == "gm":
                start = time.time()
                X_train, y_train = gradient_matching(
                    X=X_train_to_distill,
                    y=y_train,
                    N=pipeline_config.distill_size,
                    random_state=pipeline_config.random_seed,
                    **pipeline_config.distill_config.args,
                )
                time_to_distill = time.time() - start

            elif pipeline_config.distill_method_name == "forgetting":
                start = time.time()
                X_train, y_train = forgetting(
                    X=X_train_to_distill,
                    y=y_train,
                    N=pipeline_config.distill_size,
                    random_state=pipeline_config.random_seed,
                    **pipeline_config.distill_config.args,
                )
                time_to_distill = time.time() - start

            elif pipeline_config.distill_method_name == "grand":
                start = time.time()
                X_train, y_train = grand(
                    X=X_train_to_distill,
                    y=y_train,
                    N=pipeline_config.distill_size,
                    random_state=pipeline_config.random_seed,
                    **pipeline_config.distill_config.args,
                )
                time_to_distill = time.time() - start

            elif pipeline_config.distill_method_name == "glister":
                start = time.time()
                X_train, y_train = glister(
                    X=X_train_to_distill,
                    y=y_train,
                    N=pipeline_config.distill_size,
                    random_state=pipeline_config.random_seed,
                    **pipeline_config.distill_config.args,
                )
                time_to_distill = time.time() - start

            elif pipeline_config.distill_method_name == "graph_cut":
                start = time.time()
                X_train, y_train = graph_cut(
                    X=X_train_to_distill,
                    y=y_train,
                    N=pipeline_config.distill_size,
                    random_state=pipeline_config.random_seed,
                    **pipeline_config.distill_config.args,
                )
                time_to_distill = time.time() - start

            elif pipeline_config.distill_method_name == "mtt":
                start = time.time()
                X_train, y_train = trajectory_matching(
                    X=X_train_to_distill,
                    y=y_train,
                    N=pipeline_config.distill_size,
                    random_state=pipeline_config.random_seed,
                    **pipeline_config.distill_config.args,
                )
                time_to_distill = time.time() - start

            elif pipeline_config.distill_method_name == "datm":
                start = time.time()
                X_train, y_train = datm(
                    X=X_train_to_distill,
                    y=y_train,
                    N=pipeline_config.distill_size,
                    random_state=pipeline_config.random_seed,
                    **pipeline_config.distill_config.args,
                )
                time_to_distill = time.time() - start

            else:
                raise NotImplementedError(
                    f"distill method [{pipeline_config.distill_method_name}] is not implemented"
                )

            # only applicable when distilling in encoded space
            if scaler is not None:
                X_train = scaler.inverse_transform(X_train)

            # if we distill in encoded and decoding it
            if (
                pipeline_config.distill_space == "encoded"
                and pipeline_config.output_space == "decoded"
            ):
                logger.info("Decoding encoded")
                start = time.time()
                X_train = encoder.decode(torch.from_numpy(X_train).float())
                if pipeline_config.convert_binary:
                    logger.info("Converting decoded to binary")
                    X_train = encoder.convert_binary(X_train)
                X_train = X_train.detach().cpu().numpy()
                time_to_distill += time.time() - start

            # if we distill in encoded space and using idxs to find in original
            if (
                pipeline_config.distill_config.is_cluster
                and pipeline_config.cluster_center == "closest"
                and pipeline_config.output_space == "original"
            ):
                logger.info("Using clustering method indices")
                X_train, y_train = (
                    original_X_train[distilled_idxs].copy(),
                    original_y_train[distilled_idxs].copy(),
                )

            # if postprocessing is involved
            if pipeline_config.use_post_data_mode:
                logger.info(
                    f"Loading post data mode: {pipeline_config.distill_config.post_data_mode_config.identifier}"
                )
                new_dm = TabularDataModule(
                    dataset_config=pipeline_config.dataset_config,
                    data_mode_config=pipeline_config.distill_config.post_data_mode_config,
                )
                new_dm.load()

                original_X_train = new_dm.subsets["train"].X.cpu().numpy()
                original_y_train = new_dm.subsets["train"].y.cpu().numpy()

                X_val = new_dm.subsets["val"].X.cpu().numpy()
                y_val = new_dm.subsets["val"].y.cpu().numpy()
                X_test = new_dm.subsets["test"].X.cpu().numpy()
                y_test = new_dm.subsets["test"].y.cpu().numpy()
                feature_mask = new_dm.feature_mask
                feature_categ_mask = new_dm.feature_categ_mask

                # if cluster mode and using closest, we can just use the index.
                if (
                    pipeline_config.distill_config.is_cluster
                    and pipeline_config.cluster_center == "closest"
                    and distilled_idxs is not None
                ):
                    logger.info("Using clustering method indicies")
                    X_train = original_X_train[distilled_idxs]
                    y_train = original_y_train[distilled_idxs]
                else:
                    old_cont = np.array(
                        [
                            not data_module.feature_categ_mask[idx]
                            for idx in data_module.feature_mask
                        ]
                    )
                    new_cont = np.array(
                        [
                            not new_dm.feature_categ_mask[idx]
                            for idx in new_dm.feature_mask
                        ]
                    )
                    if (
                        data_module.data_mode_config.parse_mode == "onehot"
                        and new_dm.data_mode_config.parse_mode == "mixed"
                    ):
                        logger.info("Converting onehot to mixed")
                        # undo onehot, but only if we have cont. features.
                        if (~data_module.feature_categ_mask).any():
                            onehot_encoder = data_module.load_onehot_encoder()
                            mixed_output = np.zeros((X_train.shape[0], len(new_cont)))
                            mixed_output[:, ~new_cont] = X_train[:, ~old_cont]
                            mixed_output[:, new_cont] = onehot_encoder.inverse_transform(
                                X_train[:, old_cont]
                            )
                            X_train = mixed_output
                        
                    elif (
                        data_module.data_mode_config.parse_mode == "mixed"
                        and new_dm.data_mode_config.parse_mode == "onehot"
                    ):
                        logger.info("Converting mixed to onehot")
                        onehot_encoder = new_dm.load_onehot_encoder()
                        onehot_output = np.zeros((X_train.shape[0], len(new_cont)))
                        onehot_output[:, ~new_cont] = X_train[:, ~old_cont]
                        onehot_output[:, new_cont] = onehot_encoder.transform(
                            X_train[:, old_cont]
                        ).toarray()
                        X_train = onehot_output
                    else:
                        raise NotImplementedError(
                            "[{}] -> [{}] this case is not implemented".format(
                                pipeline_config.data_mode_config.parse_mode,
                                new_dm.data_mode_config.parse_mode,
                            )
                        )

            if use_cache:
                logger.info("Saving to cache")
                logger.info(str(pipeline_config.distilled_data_dir))
                pipeline_config.distilled_data_dir.parent.mkdir(
                    parents=True, exist_ok=True
                )
                with open(pipeline_config.distilled_data_dir, "wb") as f:
                    with h5py.File(f, "w") as cache_file:
                        cache_file.create_dataset("X_train", data=X_train)
                        cache_file.create_dataset("y_train", data=y_train)
                pipeline_config.save_distill_time(time_to_distill)

        else:
            with open(pipeline_config.distilled_data_dir, "rb") as f:
                with h5py.File(f, "r") as cache_file:
                    X_train = cache_file["X_train"][:]
                    y_train = cache_file["y_train"][:]
            time_to_distill = pipeline_config.load_distill_time()

            logger.info("Loading from cache")
            logger.info(str(pipeline_config.distilled_data_dir))

            # if postprocessing is involved
            if pipeline_config.use_post_data_mode:
                logger.info(
                    f"Loading post data mode: {pipeline_config.distill_config.post_data_mode_config.identifier}"
                )
                new_dm = TabularDataModule(
                    dataset_config=pipeline_config.dataset_config,
                    data_mode_config=pipeline_config.distill_config.post_data_mode_config,
                )
                new_dm.load()
                original_X_train = new_dm.subsets["train"].X.cpu().numpy()
                original_y_train = new_dm.subsets["train"].y.cpu().numpy()
                X_val = new_dm.subsets["val"].X.cpu().numpy()
                y_val = new_dm.subsets["val"].y.cpu().numpy()
                X_test = new_dm.subsets["test"].X.cpu().numpy()
                y_test = new_dm.subsets["test"].y.cpu().numpy()
                feature_mask = new_dm.feature_mask
                feature_categ_mask = new_dm.feature_categ_mask

        if (
            pipeline_config.distill_space == "encoded"
            and pipeline_config.output_space == "encoded"
        ):
            # if we are not decoding the output, turn the val and test X into the encoded space as well
            original_X_train = X_train_to_distill
            X_val = X_val_encoded
            X_test = X_test_encoded

    logger.info("Shapes")
    logger.info(f"- Train: {X_train.shape}, {X_train.dtype}")
    logger.info(f"- Train Original: {original_X_train.shape}, {original_X_train.dtype}")
    logger.info(f"- Val: {X_val.shape}, {X_val.dtype}")
    logger.info(f"- Test: {X_test.shape}, {X_test.dtype}")
    logger.info("-" * 40)

    if X_train.shape[1] != original_X_train.shape[1]:
        raise Exception(
            f"X_train and original_X_train have different number of features {X_train.shape[1]} != {original_X_train.shape[1]}"
        )
    if X_train.shape[1] != X_val.shape[1]:
        raise Exception(
            f"X_train and X_val have different number of features {X_train.shape[1]} != {X_val.shape[1]}"
        )
    if X_train.shape[1] != X_test.shape[1]:
        raise Exception(
            f"X_train and X_test have different number of features {X_train.shape[1]} != {X_test.shape[1]}"
        )

    distilled_data = {
        "train": TabularDataset(
            X=X_train,
            y=y_train,
            feature_mask=feature_mask,
            feature_categ_mask=feature_categ_mask,
        ),
        "train - original": TabularDataset(
            X=original_X_train,
            y=original_y_train,
            feature_mask=feature_mask,
            feature_categ_mask=feature_categ_mask,
        ),
        "val": TabularDataset(
            X=X_val,
            y=y_val,
            feature_mask=feature_mask,
            feature_categ_mask=feature_categ_mask,
        ),
        "test": TabularDataset(
            X=X_test,
            y=y_test,
            feature_mask=feature_mask,
            feature_categ_mask=feature_categ_mask,
        ),
    }

    # this probably doesn't do much, but we want to save as much memory as possible.
    del data_module
    return distilled_data
