import hydra
import torch
import pandas as pd
import json
from omegaconf import DictConfig, OmegaConf
from rich.progress import Progress

from tabdd.data import TabularDataModule
from tabdd.models.encoder import load_encoder
from tabdd.models.utils import metric
from tabdd.config import (
    DatasetConfig,
    DataModeConfig,
    EncoderTuneConfig,
    EncoderTrainConfig,
    MultiEncoderTuneConfig,
)
from tabdd.config.paths import RESULTS_CACHE_DIR, RUN_CONFIG_DIR
from tabdd.utils import progress_bar
from tabdd.config import (
    load_data_mode_config,
    load_dataset_configs,
    load_encoder_tune_configs,
    load_encoder_train_config,
)


def evaluate_encoder(
    dataset_config: DatasetConfig,
    data_mode_config: DataModeConfig,
    encoder_tune_config: EncoderTuneConfig | MultiEncoderTuneConfig,
    encoder_train_config: EncoderTrainConfig,
    overwrite: bool = False,
):
    dm = TabularDataModule(
        dataset_config=dataset_config,
        data_mode_config=data_mode_config,
    )

    is_multihead = isinstance(encoder_tune_config, MultiEncoderTuneConfig)

    best_checkpoints_dir = (
        encoder_train_config.checkpoint_dir
        / encoder_tune_config.identifier
        / dm.identifier
        / "asha_hyperopt"
        / str(encoder_train_config.latent_dim)
    )

    metric_file = best_checkpoints_dir / "final_metrics.json"
    dm.prepare_data()
    dm.setup()

    device = "cpu"
    if torch.cuda.is_available():
        device = torch.device(0)

    if overwrite or not metric_file.is_file():

        encoder, best_checkpoint = load_encoder(
            encoder_tune_config=encoder_tune_config,
            encoder_train_config=encoder_train_config,
            data_module=dm,
            device=device,
        )

        encoder.eval()

        with torch.no_grad():
            if encoder is None:
                return None

            result = {}

            with open(best_checkpoint / "runtime.json") as f:
                runtime = json.load(f)
                result["Ray Tune Runtime"] = runtime["ray_run_time"]
                result["Total Tune Runtime"] = sum(runtime["per_run_time_indiv"])

            if not is_multihead:
                for subset_name, subset in dm.subsets.items():
                    result[f"{subset_name.capitalize()} Recon Accuracy"] = metric(
                        encoder_train_config.target_metric,
                        subset.X.to(device),
                        encoder.decode(
                            encoder.encode(
                                subset.X.to(device),
                                feature_idx=subset.feature_idx.to(device),
                            )
                        )
                        .detach()
                        .cpu(),
                        feature_mask=encoder.feature_mask.to(device),
                    )
                result["Encoder Params"] = sum(
                    [p.numel() for p in encoder.encoder.parameters() if p.requires_grad]
                )
                result["Decoder Params"] = sum(
                    [p.numel() for p in encoder.decoder.parameters() if p.requires_grad]
                )
            else:
                for subset_name, subset in dm.subsets.items():
                    result[f"{subset_name.capitalize()} Recon Accuracy"] = metric(
                        encoder_train_config.autoencoder_target_metric,
                        subset.X.to(device),
                        encoder.decode(
                            encoder.encode(
                                subset.X.to(device),
                                feature_idx=subset.feature_idx.to(device),
                            )
                        )
                        .detach()
                        .cpu(),
                        feature_mask=encoder.feature_mask.to(device),
                    )

                for subset_name, subset in dm.subsets.items():
                    result[f"{subset_name.capitalize()} Predict Accuracy"] = metric(
                        encoder_train_config.classifier_target_metric,
                        subset.y,
                        encoder.classifier(
                            encoder.encode(
                                subset.X.to(device),
                                feature_idx=subset.feature_idx.to(device),
                            )
                        )
                        .detach()
                        .cpu(),
                        num_classes=dm.dataset_config.n_classes,
                    )

                result["Encoder Params"] = sum(
                    [
                        p.numel()
                        for p in encoder.autoencoder.encoder.parameters()
                        if p.requires_grad
                    ]
                )
                result["Decoder Params"] = sum(
                    [
                        p.numel()
                        for p in encoder.autoencoder.decoder.parameters()
                        if p.requires_grad
                    ]
                )
                result["Classifier Params"] = sum(
                    [
                        p.numel()
                        for p in encoder.classifier.parameters()
                        if p.requires_grad
                    ]
                )

        json.dump(result, open(metric_file, "w"))
    else:
        result = json.load(open(metric_file, "r"))

    return result


def load_enc_stats(
    dataset_config: DatasetConfig,
    data_mode_config: DataModeConfig,
    encoder_tune_configs: list[EncoderTuneConfig | MultiEncoderTuneConfig],
    encoder_train_config: EncoderTrainConfig,
    progress: Progress = None,
    overwrite: bool = False,
) -> pd.DataFrame:
    reports = []

    if progress is not None:
        encoder_task = progress.add_task("Encoder", total=len(encoder_tune_configs))

    for encoder_tune_config in sorted(
        encoder_tune_configs, key=lambda x: x.pretty_name
    ):
        report = evaluate_encoder(
            dataset_config=dataset_config,
            data_mode_config=data_mode_config,
            encoder_tune_config=encoder_tune_config,
            encoder_train_config=encoder_train_config,
            overwrite=overwrite,
        )
        if report is not None:
            reports.append(
                {
                    "Model": encoder_tune_config.pretty_name,
                    "Dataset": dataset_config.dataset_name,
                    "Data Parse Mode": data_mode_config.parse_mode,
                    **report,
                }
            )
        if progress is not None:
            progress.update(encoder_task, advance=1)
    if progress is not None:
        progress.remove_task(encoder_task)

    reports = pd.DataFrame(reports)
    return reports


def load_all_enc_stats(config: DictConfig, overwrite: bool = False) -> pd.DataFrame:
    data_mode_config = load_data_mode_config(config)
    dataset_configs = load_dataset_configs(config)
    encoder_tune_configs = load_encoder_tune_configs(config)
    encoder_train_config = load_encoder_train_config(config)

    # vanilla_train_conf.update(encoder_train_config)
    # multi_train_conf.update(encoder_train_config)

    report = []
    with progress_bar() as pbar:
        dataset_task = pbar.add_task("Dataset", total=len(dataset_configs))
        for dataset_config in dataset_configs:
            report += [
                load_enc_stats(
                    dataset_config=dataset_config,
                    data_mode_config=data_mode_config,
                    encoder_tune_configs=encoder_tune_configs,
                    encoder_train_config=encoder_train_config,
                    progress=pbar,
                    overwrite=overwrite,
                ),
            ]
            pbar.update(dataset_task, advance=1)
    report = pd.concat(report)

    return report


@hydra.main(version_base=None, config_path=RUN_CONFIG_DIR, config_name="all")
def run(config: DictConfig) -> None:
    load_all_enc_stats(config)


if __name__ == "__main__":
    run()
