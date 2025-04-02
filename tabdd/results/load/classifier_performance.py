import hydra
from typing import Literal, Callable
from omegaconf import DictConfig
import pandas as pd
import numpy as np
from tabdd.utils import progress_bar

from tabdd.config.paths import RUN_CONFIG_DIR, DATA_REPO_DIR
from tabdd.config import (
    DatasetConfig,
    EncoderTuneConfig,
    EncoderTrainConfig,
    ClassifierTuneConfig,
    ClassifierTrainConfig,
    DistillConfig,
    DataModeConfig,
    load_data_mode_config,
    load_dataset_configs,
    load_encoder_train_config,
    load_encoder_tune_configs,
    load_distill_configs,
    load_classifier_tune_configs,
    load_classifier_train_config,
    load_pipeline_configs,
)
from tabdd.utils import progress_bar


def data_mode_acronyms(data_mode):
    return (
        data_mode.replace("Agglo", "AG")
        .replace("MLP-MultiHead", "[FFN-SFT]")
        .replace("MLP", "FFN")
        .replace("GNN-MultiHead", "[GNN-SFT]")
        .replace("TF-MultiHead", "[TF-SFT]")
        .replace("Encoded", "ENC")
        .replace("Decoded", "DEC")
        .replace("Original", "ORI")
        .replace("Random Sample", "RS")
        .replace("Binary", "BIN")
        .replace(" / Centroid", "/SYN")
        .replace(" / Closest", "/REAL")
    )


def compare_group_key(group: pd.DataFrame, metric: str, strategy: str):
    if strategy == "max":
        return group[metric].max()
    elif strategy == "min":
        return group[metric].min()
    elif strategy == "mean":
        return group[metric].mean()
    elif isinstance(strategy, Callable):
        return strategy(group, metric)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")


def aggregate_perf(group, metric):
    return {
        f"{metric} Values": group[metric].tolist(),
        f"{metric} Mean": group[metric].mean(),
        f"{metric} Max": group[metric].max(),
        f"{metric} Min": group[metric].min(),
    }


def compute_groups(
    report: pd.DataFrame,
    targets: list[str],
    metric: str,
    exclude: list[str] = [],
    group_aggr: str = "mean",
) -> pd.DataFrame:

    # Base filter so we only look at the test set
    base_filter = report["Subset"] == "Test"

    # The rank group must at least be this large. if not we skip b/c we dont' have a complete rank
    min_group_size = report[targets].drop_duplicates().shape[0]

    # gather columns that represent the pipeline components
    other_components = list(
        set(report.columns)
        - set(
            [
                "Subset",
                "Score",
                "Default Train Time",
                "Opt Train Time",
                "Opt Train Time Total",
                "Inference Time",
                "Data Distill Time",
                "Data Mode",
                "Short Name",
                "Regret",
                "Scaled Regret",
                "Scaled Regret with RS",
            ]
        )
        - set(targets)
        - set(exclude)
    )

    # Save methods that actually do distillation
    distill_methods = set(report["Distill Method"].unique()) - set(
        ["Original", "Encoded", "Decoded", "Random Sample"]
    )

    targets_filter = True
    if "Distill Method" in targets:
        targets_filter &= report["Distill Method"].isin(distill_methods)

    filtered = report[base_filter & targets_filter].copy()

    print(f"Grouping {targets} by {metric}, grouped by: {other_components}")

    all_groups = []
    with progress_bar() as progress:

        # if distill method is in the targets, we will pre-compute the special cases
        grouped = list(filtered.groupby(other_components))
        main_task = progress.add_task("Ranking", total=len(grouped))
        for gr_idx, (comps, rank_group) in enumerate(grouped):
            comps = dict(zip(other_components, comps))

            # if the rank group's targets are less than min_group_size, runs are not complete yet.
            if rank_group[targets].drop_duplicates().shape[0] < min_group_size:
                progress.update(main_task, advance=1)
                continue

            # take average
            all_groups += [
                {
                    "Target": targ,
                    "Metric": compare_group_key(
                        group=perf_per_dm,
                        metric=metric,
                        strategy=group_aggr,
                    ),
                    "Group ID": gr_idx,
                    **aggregate_perf(perf_per_dm, "Score"),
                    **aggregate_perf(perf_per_dm, "Regret"),
                    **aggregate_perf(perf_per_dm, "Scaled Regret"),
                    **aggregate_perf(perf_per_dm, "Scaled Regret with RS"),
                    **dict(zip(targets, targ)),
                }
                for targ, perf_per_dm in rank_group.groupby(targets)
            ]
            progress.update(main_task, advance=1)
            # all_groups.append(rankings)
        progress.remove_task(main_task)
    all_groups = pd.DataFrame(all_groups)
    print("Computing Ranks")

    return all_groups


def compute_ranks(
    groups,
    direction: Literal["max", "min"] = "max",
    target: str = "Target",
    metric: str = "Metric",
):
    rankings = pd.DataFrame(
        [
            dict(
                zip(
                    rank_group[target],
                    rank_group[metric].rank(ascending=(direction != "max")) - 1,
                )
            )
            for _, rank_group in groups.groupby("Group ID")
        ]
    )
    return rankings


def compute_regret(raw_results: pd.DataFrame) -> pd.DataFrame:
    is_distill_pipeline = ~raw_results["Distill Method"].isin(
        ["Original", "Encoded", "Decoded"]
    )

    # need to duplicate baselines for each distill size
    results = pd.concat(
        [
            pd.concat(
                [
                    raw_results[~is_distill_pipeline].assign(**{"N": n})
                    for n in sorted(set(raw_results["N"].unique()) - set([0]))
                ]
            ),
            raw_results[is_distill_pipeline],
        ]
    )

    # calculate regret score
    results["Balanced Regret"] = -np.inf
    results["Regret"] = -np.inf
    for n, grouped in results[results["Subset"] == "Test"].groupby("N"):
        if "original" in grouped["Distill Method"].values:
            original_score = grouped[(grouped["Data Mode"] == "Mixed Original")][
                "Score"
            ].mean()
            results.loc[
                ((results["N"] == n) & (results["Subset"] == "Test")), "Regret"
            ] = (original_score - grouped["Score"]) / original_score

            if "Random Sample" in grouped["Distill Method"].values:
                random_score = grouped[(grouped["Data Mode"] == "Random Sample")][
                    "Score"
                ].mean()
                results.loc[
                    (results["N"] == n) & (results["Subset"] == "Test"),
                    "Balanced Regret",
                ] = (original_score - grouped["Score"]) / (
                    original_score - random_score + 1e-10
                )

    return results


def load_clf_perf(
    dataset_config: DatasetConfig,
    data_mode_config: DataModeConfig,
    classifier_tune_config: ClassifierTuneConfig,
    classifier_train_config: ClassifierTrainConfig,
    encoder_tune_config: EncoderTuneConfig | None,
    encoder_train_config: EncoderTrainConfig,
    distill_config: DistillConfig,
    refresh: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:

    save_dir = (
        classifier_train_config.results_dir
        / "reports"
        / dataset_config.dataset_name
        / (
            f"{data_mode_config.identifier}/{encoder_train_config.identifier}/{encoder_tune_config.identifier}"
            if encoder_tune_config is not None
            else data_mode_config.identifier
        )
        / distill_config.identifier
        / classifier_tune_config.identifier
        / classifier_train_config.identifier
    )

    report_file = save_dir / "complete_report.csv"

    pipeline_configs = load_pipeline_configs(
        dataset_config=dataset_config,
        data_mode_config=data_mode_config,
        classifier_tune_config=classifier_tune_config,
        classifier_train_config=classifier_train_config,
        distill_config=distill_config,
        encoder_tune_configs=(
            [] if encoder_tune_config is None else [encoder_tune_config]
        ),
        encoder_train_config=encoder_train_config,
    )
    pipeline_attributes = list(pipeline_configs[0].attributes.keys())

    empty_report = pd.DataFrame(
        [],
        columns=[
            "Subset",
            "Score",
            "Balanced Regret",
            "Regret",
        ]
        + pipeline_attributes,
    )
    empty_incomplete = pd.DataFrame([], columns=pipeline_attributes)

    if refresh or (not report_file.is_file()):
        raw_results = []
        incomplete_runs = []

        for pipeline_config in pipeline_configs:
            config_params = pipeline_config.attributes
            # print(pipeline_config.report_dir, pipeline_config.is_complete)
            if pipeline_config.is_complete:
                report = pipeline_config.load_report_w_runtime()
                for k, v in config_params.items():
                    report[k] = v
                raw_results.append(report)
            else:
                incomplete_runs.append(config_params)

        if len(raw_results) == 0:
            return empty_report, empty_incomplete

        raw_results = pd.concat(raw_results).reset_index(drop=True)
        incomplete_runs = pd.DataFrame(incomplete_runs)

        raw_results["Short Name"] = raw_results["Data Mode"].apply(data_mode_acronyms)

        # if we are done, then save the results
        if len(incomplete_runs) == 0:
            incomplete_runs = empty_incomplete
            save_dir.mkdir(parents=True, exist_ok=True)
            raw_results.to_csv(report_file, index=False)

    else:
        # pandas treats "N/A" as nan... so we need to manually fill out missing fields
        raw_results = pd.read_csv(report_file).fillna("N/A")
        incomplete_runs = empty_incomplete

    return raw_results, incomplete_runs


def load_all_clf_perf(
    config: DictConfig,
    verbose: bool = False,
    refresh: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    data_mode_config = load_data_mode_config(config)
    dataset_configs = load_dataset_configs(config)
    encoder_tune_configs = load_encoder_tune_configs(config)
    encoder_train_config = load_encoder_train_config(config)
    distill_configs = load_distill_configs(config)
    classifier_tune_configs = load_classifier_tune_configs(config)
    classifier_train_config = load_classifier_train_config(config)

    if len(encoder_tune_configs) == 0:
        encoder_tune_configs = [None]

    results = []
    incomplete = []

    if verbose:
        print("=" * 40)
        print("Loading all reports in config...")
        print()
    with progress_bar() as progress:
        ds_task = progress.add_task("Datasets ..", total=len(dataset_configs))
        for i_ds, dataset_config in enumerate(dataset_configs):
            res_per_ds = []

            clf_task = progress.add_task(
                "Classifiers ..", total=len(classifier_tune_configs)
            )
            for i_c, classifier_tune_config in enumerate(classifier_tune_configs):
                enc_task = progress.add_task(
                    "Encoders ..", total=len(encoder_tune_configs)
                )
                for i_e, encoder_tune_config in enumerate(encoder_tune_configs):
                    dd_task = progress.add_task(
                        "Distill ..", total=len(distill_configs)
                    )
                    for i_d, distill_config in enumerate(distill_configs):
                        result, incomplete_runs = load_clf_perf(
                            dataset_config=dataset_config,
                            data_mode_config=data_mode_config,
                            classifier_tune_config=classifier_tune_config,
                            classifier_train_config=classifier_train_config,
                            distill_config=distill_config,
                            encoder_train_config=encoder_train_config,
                            encoder_tune_config=encoder_tune_config,
                            refresh=refresh,
                        )

                        if verbose:
                            print("-" * 40)
                            print("Loaded")
                            print(
                                f"{i_ds+1}/{len(dataset_configs):<3} -- {dataset_config.dataset_name}"
                            )
                            print(
                                f"{i_c+1}/{len(classifier_tune_configs):<3} -- {classifier_tune_config.classifier_name}"
                            )
                            print(
                                f"{i_e+1}/{len(encoder_tune_configs):<3} -- {encoder_tune_config.pretty_name}"
                            )
                            print(
                                f"{i_d+1}/{len(distill_configs):<3} -- {distill_config.pretty_name}"
                            )

                            if (len(incomplete_runs) == 0) and (len(results) > 0):
                                print("Complete!")
                            print("-" * 40)

                        res_per_ds.append(result)
                        incomplete.append(incomplete_runs)
                        progress.advance(dd_task, 1)
                    progress.remove_task(dd_task)
                    progress.advance(enc_task, 1)
                progress.remove_task(enc_task)
                progress.advance(clf_task, 1)
            progress.remove_task(clf_task)
            progress.advance(ds_task, 1)

            filtered = [r for r in res_per_ds if len(r)]

            if not len(filtered) > 0:
                continue

            res_per_ds = pd.concat(filtered)

            if res_per_ds.isna().any().any():
                print("### ISSUE ###")
                print(f" there are nans.. :{res_per_ds.isna().any()}")

            if res_per_ds["Data Distill Time"].isna().any():
                print("distill time nans...")
                print(
                    res_per_ds[res_per_ds["Data Distill Time"].isna()][
                        ["Data Distill Time", "Data Mode"]
                    ]
                )
                quit()

            # drop duplicates that do not depend on a encoder
            no_encoder = res_per_ds[res_per_ds["Encoder"] == ""].drop_duplicates()
            uses_encoder = res_per_ds[res_per_ds["Encoder"] != ""]

            to_concat = []
            if len(no_encoder):
                to_concat.append(no_encoder)
            if len(uses_encoder):
                to_concat.append(uses_encoder)

            res_per_ds = pd.concat(to_concat)
            # print("we concated separated by encoder")

            is_distill_pipeline = ~res_per_ds["Distill Method"].isin(
                ["Original", "Encoded", "Decoded"]
            )

            # need to duplicate baselines for each distill size
            ns_to_fill = sorted(set(res_per_ds["N"].unique()) - set([0])) 
            if len(ns_to_fill) > 0:
                res_w_baseline = pd.concat(
                    [
                        pd.concat(
                            [
                                res_per_ds[~is_distill_pipeline].assign(**{"N": n})
                                for n in ns_to_fill
                            ]
                        ),
                        res_per_ds[is_distill_pipeline],
                    ]
                )
            else:
                res_w_baseline = res_per_ds

            # res_w_reg = compute_regret(res_per_ds)
            results.append(res_w_baseline)

    if verbose:
        print()
        print("=" * 40)
        print("Loaded all.")

    results = pd.concat(results)
    incomplete = pd.concat(incomplete)

    results_datasets = set(results["Dataset"].unique())
    incomplete_datasets = set([])

    if len(incomplete) > 0:
        incomplete_datasets = set(incomplete["Dataset"].unique())
    print(
        f"{len(results_datasets-incomplete_datasets)}/{len(results_datasets|incomplete_datasets)} datasets complete!"
    )
    if len(incomplete_datasets) > 0:
        print("Incomplete datasets:")
        for d in sorted(incomplete_datasets):
            print(f"    - {d}")

    return results, incomplete


@hydra.main(version_base=None, config_path=RUN_CONFIG_DIR, config_name="all")
def run(config: DictConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    results, incomplete = load_all_clf_perf(config, verbose=True, refresh=True)


if __name__ == "__main__":
    run()
