import hydra
from omegaconf import DictConfig
import torch
import pytorch_lightning as pl
import ray
from ray import tune, air
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.hyperopt import HyperOptSearch

# from ray.tune.integration.pytorch_lightning import TuneReportCheckpointCallback
from ray.tune.integration.pytorch_lightning import (
    TuneReportCheckpointCallback,
    TuneReportCallback,
)
import math

from pathlib import Path
import shutil
import json
from datetime import datetime
import os
import time
import pandas as pd

from tabdd.data import TabularDataModule
from tabdd.models.encoder import MultiHeadAutoEncoder, load_encoder
from tabdd.config import (
    load_data_mode_config,
    load_dataset_configs,
    load_encoder_train_config,
    load_encoder_tune_configs,
    EncoderTuneConfig,
    MultiEncoderTuneConfig,
    EncoderTrainConfig,
    DatasetConfig,
    DataModeConfig,
)
from tabdd.config.paths import (
    RAY_TMP_DIR,
    RAY_RESULT_DIR,
    RUN_CONFIG_DIR,
)


def run_one(
    dataset_config: DatasetConfig,
    data_mode_config: DataModeConfig,
    encoder_train_config: EncoderTrainConfig,
    encoder_tune_config: EncoderTuneConfig | MultiEncoderTuneConfig,
):
    use_gpu = encoder_train_config.gpu_per_worker > 0

    resources_per_worker = (
        {
            "CPU": encoder_train_config.cpu_per_worker,
            "GPU": encoder_train_config.gpu_per_worker,
        }
        if use_gpu
        else {"CPU": encoder_train_config.cpu_per_worker}
    )

    target_metric = (
        encoder_train_config.autoencoder_target_metric
        if isinstance(encoder_tune_config, EncoderTuneConfig)
        else encoder_train_config.target_metric
    )

    def train(train_config):
        dm = TabularDataModule(
            dataset_config=dataset_config,
            data_mode_config=data_mode_config,
        )
        dm.prepare_data()

        model = None

        if isinstance(encoder_tune_config, EncoderTuneConfig):
            model = encoder_tune_config.cls(
                **train_config,
                latent_dim=encoder_train_config.latent_dim,
                criterion=encoder_train_config.criterion,
                metrics=[encoder_train_config.autoencoder_target_metric],
                in_dim=dm.x_dim,
                feature_mask=dm.feature_mask,
                feature_categ_mask=dm.feature_categ_mask,
            )
        else:
            base_autoencoder, _ = load_encoder(
                encoder_tune_config=encoder_tune_config.base_encoder_config,
                encoder_train_config=encoder_train_config,
                data_module=dm,
            )

            model = MultiHeadAutoEncoder(
                **train_config,
                autoencoder=base_autoencoder,
                latent_dim=encoder_train_config.latent_dim,
                autoencoder_criterion=encoder_train_config.autoencoder_criterion,
                autoencoder_metrics=[encoder_train_config.autoencoder_target_metric],
                classifier_criterion=encoder_train_config.classifier_criterion,
                classifier_metrics=[encoder_train_config.classifier_target_metric],
                combined_metric_balance=encoder_train_config.combined_metric_balance,
                in_dim=dm.x_dim,
                out_dim=dm.dataset_config.n_classes,
                feature_mask=dm.feature_mask,
                feature_categ_mask=dm.feature_categ_mask,
            )

        trainer = pl.Trainer(
            logger=True,
            enable_checkpointing=False,
            accelerator="auto" if use_gpu else "cpu",
            enable_progress_bar=False,
            max_epochs=encoder_train_config.num_epochs,
            callbacks=[
                TuneReportCheckpointCallback(
                    {
                        f"train/{target_metric}": f"train/{target_metric}",
                        f"val/{target_metric}": f"val/{target_metric}",
                    },
                    on="validation_end",
                )
            ],
        )
        trainer.fit(
            model=model,
            datamodule=dm,
        )

    scheduler = ASHAScheduler(
        max_t=encoder_train_config.num_epochs,
        grace_period=1,
        reduction_factor=2,
    )

    search_alg = HyperOptSearch(
        metric=f"val/{target_metric}",
        mode="max",
        random_state_seed=0,
    )

    tuner = tune.Tuner(
        tune.with_resources(
            tune.with_parameters(
                train,
            ),
            resources=resources_per_worker,
        ),
        tune_config=tune.TuneConfig(
            metric=f"val/{target_metric}",
            mode="max",
            num_samples=encoder_train_config.num_samples,
            scheduler=scheduler,
            search_alg=search_alg,
            reuse_actors=True,
            max_concurrent_trials=encoder_train_config.max_concurrent_trials,
        ),
        run_config=air.RunConfig(
            checkpoint_config=air.CheckpointConfig(
                num_to_keep=1,
                checkpoint_score_attribute=f"val/{target_metric}",
                checkpoint_score_order="max",
            ),
            storage_path=RAY_RESULT_DIR,
            name="{}/{}/{}/asha_hyperopt/{}".format(
                encoder_tune_config.identifier,
                dataset_config.dataset_name,
                data_mode_config.identifier,
                encoder_train_config.latent_dim,
            ),
        ),
        param_space=encoder_tune_config.tune_params,
    )

    print("everything is ready. starting tuning...")
    start = time.time()
    results = tuner.fit()
    end = time.time()

    print("we are done!")

    best_result = results.get_best_result(metric=f"val/{target_metric}", mode="max")

    print("=" * 80)
    print("Best result")
    print(best_result)
    print(best_result.config)

    best_checkpoint = Path(best_result.checkpoint.path) / "checkpoint"
    best_config = best_result.config
    best_checkpoint_dir = (
        encoder_train_config.checkpoint_dir
        / encoder_tune_config.identifier
        / dataset_config.dataset_name
        / data_mode_config.identifier
        / "asha_hyperopt"
        / str(encoder_train_config.latent_dim)
        / str(datetime.now())
    )
    best_checkpoint_dir.mkdir(exist_ok=True, parents=True)

    shutil.copy(best_checkpoint, best_checkpoint_dir / "model_weight.pkl")
    with open(best_checkpoint_dir / "model_config.json", "w") as f:
        json.dump(best_config, f, indent=2)
    best_result.metrics_dataframe.to_csv(
        best_checkpoint_dir / "metrics.csv", index=False
    )
    print(f"best results saved in {best_checkpoint_dir.resolve()}/metrics.csv")
    print(best_result.metrics_dataframe.iloc[-1:, :2])

    ray.shutdown()

    per_run_times = []
    for r in results:
        logfile = r.log_dir / "progress.csv"
        if not logfile.exists():
            continue
        per_run_times.append(pd.read_csv(logfile)["time_this_iter_s"].sum())

    json.dump(
        {
            "ray_run_time": end - start,
            "per_run_time": sum(per_run_times),
            "per_run_time_indiv": per_run_times,
        },
        open(best_checkpoint_dir / "runtime.json", "w"),
    )


@hydra.main(
    version_base=None,
    config_path=RUN_CONFIG_DIR,
    config_name="tune",
)
def run(config: DictConfig):
    pl.seed_everything(0)

    dataset_configs = load_dataset_configs(config)
    data_mode_config = load_data_mode_config(config)
    encoder_train_config = load_encoder_train_config(config)
    encoder_tune_configs = load_encoder_tune_configs(config)

    ray_args = {
        "address": "local",
        "num_cpus": math.ceil(
            encoder_train_config.max_concurrent_trials
            * encoder_train_config.cpu_per_worker
        ),
    }
    if torch.cuda.is_available():
        ray_args["num_gpus"] = torch.cuda.device_count()

    if RAY_TMP_DIR is not None:
        os.environ["RAY_TMPDIR"] = RAY_TMP_DIR
        ray_args["_temp_dir"] = RAY_TMP_DIR

    ray.init(**ray_args)

    print(f"ray is up and running with {ray_args['num_gpus']} GPUs")

    for dataset_config in dataset_configs:
        for encoder_tune_config in encoder_tune_configs:
            run_one(
                dataset_config=dataset_config,
                data_mode_config=data_mode_config,
                encoder_train_config=encoder_train_config,
                encoder_tune_config=encoder_tune_config,
            )


if __name__ == "__main__":
    run()
