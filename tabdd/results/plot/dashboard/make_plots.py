import pandas as pd
import json
from pathlib import Path
from rich.progress import Progress
from argparse import ArgumentParser

from tabdd.config.paths import PLOTS_CACHE_DIR, PLOTS_DIR, DATA_DIR
from tabdd.data.load import load_openml_dataset
# from tabdd.data.data_modules import DATASETS
from tabdd.utils import progress_bar

from .plot.colors import (
    unique_colors,
    dataset_stats_colormap,
    distill_results_colormap,
    encoder_stats_colormap,
)

from .load import (
    load_classifier_performance,
    load_dataset_stats,
    load_distilled_embeddings,
    load_encoded_embeddings,
    load_enc_stats,
    load_feature_importances,
    load_tuned_parameters,
)

from .plot import (
    plot_autoencoder_stats,
    plot_baseline_performance,
    plot_classifier_performance,
    plot_classifier_performance_overall,
    plot_dataset_stats,
    plot_distilled_distribution,
    plot_encoded_distribution,
    plot_feature_distribution,
    plot_feature_importances,
    plot_multihead_autoencoder_stats,
    plot_tuned_parameters,
)

from .plot.config import (
    REDUCE_METHODS,
    AUTOENCODERS,
    DISTILL_METHODS,
    CLUSTER_CENTERS,
    MULTIHEAD_AUTOENCODERS,
    CLASSIFIERS,
)


def make_distill_results_plot(
    dataset_name: str,
    classifier_name: str,
    tune_hyperopt: bool = False,
):
    train_mode = "Default"
    if tune_hyperopt:
        train_mode = "Tuned"

    plot_file = (
        Path(PLOTS_DIR)
        / f"{dataset_name}_{classifier_name}_{train_mode}_distill_result.json"
    )

    distill_results, all_done = load_classifier_performance(
        dataset_name=dataset_name,
        classifier_name=classifier_name,
        tune_hyperopt=tune_hyperopt,
    )

    if len(distill_results["Subset"].unique()) == 0:
        return

    if distill_results is not None:
        fig = json.loads(
            plot_classifier_performance(
                distill_results=distill_results,
                title=f"{dataset_name} - {classifier_name} / {train_mode} distillation results",
                colormap=distill_results_colormap,
            ).to_json()
        )
        json.dump(fig, open(plot_file, "w"))


def make_overall_results_plot(
    classifier_name: str,
    tune_hyperopt: bool = False,
):
    train_mode = "Default"
    if tune_hyperopt:
        train_mode = "Tuned"

    plot_file = (
        Path(PLOTS_DIR) / f"{classifier_name}_{train_mode}_distill_result_overall.json"
    )

    overall_distill_results = []
    for ds in DATASETS:
        distill_results, incomplete_runs = load_classifier_performance(
            dataset_name=ds,
            classifier_name=classifier_name,
            tune_hyperopt=tune_hyperopt,
        )
        if not incomplete_runs:
            overall_distill_results += [distill_results]
        else:
            print(f"{ds} {classifier_name} Tune={tune_hyperopt} is not complete")

    if len(overall_distill_results) > 0:
        overall_distill_results = pd.concat(overall_distill_results)
        fig = json.loads(
            plot_classifier_performance_overall(
                distill_results=overall_distill_results,
                title=f"{classifier_name} / {train_mode} performance on distilled data over datasets",
                colormap=distill_results_colormap,
            ).to_json()
        )
        json.dump(fig, open(plot_file, "w"))


def make_baseline_results_plot(
    dataset_name: str, classifier_name: str, tune_hyperopt: bool = False
):
    train_mode = "Default"
    if tune_hyperopt:
        train_mode = "Tuned"

    plot_file = (
        Path(PLOTS_DIR)
        / f"{dataset_name}_{classifier_name}_{train_mode}_baseline_results.json"
    )

    distill_results, all_done = load_classifier_performance(
        dataset_name=dataset_name,
        classifier_name=classifier_name,
        tune_hyperopt=tune_hyperopt,
    )

    if distill_results is not None:
        fig = json.loads(
            plot_baseline_performance(
                distill_results=distill_results,
                title=f"{dataset_name} - {classifier_name} / {train_mode} performance on baseline data",
                colormap=distill_results_colormap,
            ).to_json()
        )
        json.dump(fig, open(plot_file, "w"))


def make_tuned_parameters_plot(
    dataset_name: str,
    classifier_name: str,
):
    plot_file = (
        Path(PLOTS_DIR) / f"{dataset_name}_{classifier_name}_tuned_parameters.json"
    )

    tuned_parameters, all_done = load_tuned_parameters(
        dataset_name, "balanced_accuracy", classifier_name
    )
    if tuned_parameters is not None:
        fig = json.loads(
            plot_tuned_parameters(
                tuned_parameters=tuned_parameters,
                title=f"{classifier_name} tuned parameters",
                colormap=distill_results_colormap,
            ).to_json()
        )
        json.dump(fig, open(plot_file, "w"))


def make_encoded_distribution_plot(
    dataset_name: str,
    encoder_name: str,
    reduce_method: str,
):
    reduced_embs_autoencoder = load_encoded_embeddings(
        dataset_name=dataset_name,
        reduce_method=reduce_method,
        encoder_name=encoder_name,
        base_sample_size=1000,
        cache_dir=Path(PLOTS_CACHE_DIR) / "encoded_distribution",
    )
    reduced_embs_multihead = load_encoded_embeddings(
        dataset_name=dataset_name,
        reduce_method=reduce_method,
        encoder_name=f"MultiHead{encoder_name}",
        base_sample_size=1000,
        cache_dir=Path(PLOTS_CACHE_DIR) / "encoded_distribution",
    )
    reduced_embs_multihead = reduced_embs_multihead[
        reduced_embs_multihead["Space"] != "Original"
    ]
    reduced_embs_multihead["Space"] = "MultiHead " + reduced_embs_multihead["Space"]
    reduced_embs = pd.concat([reduced_embs_autoencoder, reduced_embs_multihead])
    fig = plot_encoded_distribution(
        reduced_embs=reduced_embs,
        reduce_method=reduce_method,
        encoder_name=encoder_name,
        colormap=dataset_stats_colormap["labels"],
    )
    fig.write_json(
        Path(PLOTS_DIR)
        / f"{dataset_name}_{encoder_name}_{reduce_method}_encoded_distribution.json"
    )


def make_distilled_distribution_plot(
    dataset_name: str,
    distill_method: str,
    cluster_center: str,
    reduce_method: str,
    progress: Progress = None,
):
    task = None
    if progress is not None:
        task = progress.add_task(
            f"[cyan]{dataset_name} {distill_method} {cluster_center} {reduce_method} ",
            total=10,
        )
    # if distill_method == 'kip' and cluster_center == 'closest': return
    distilled_distribution = {}
    for N in range(10, 101, 10):
        distill_results = pd.concat(
            [
                pd.concat(
                    [
                        load_distilled_embeddings(
                            dataset_name=dataset_name,
                            reduce_method=reduce_method,
                            distill_method="random_sample",
                            distill_space="original",
                            output_space="",
                            convert_binary=False,
                            cluster_center=cluster_center,
                            encoder_name="Original",
                            distill_size=N,
                            cache_dir=Path(PLOTS_CACHE_DIR) / "distilled_distribution",
                        ).assign(**{"Encoder": en}),
                        load_distilled_embeddings(
                            dataset_name=dataset_name,
                            reduce_method=reduce_method,
                            distill_method=distill_method,
                            distill_space="original",
                            output_space="encoded",
                            convert_binary=False,
                            cluster_center=cluster_center,
                            encoder_name=en,
                            distill_size=N,
                            cache_dir=Path(PLOTS_CACHE_DIR) / "distilled_distribution",
                        ),
                        load_distilled_embeddings(
                            dataset_name=dataset_name,
                            reduce_method=reduce_method,
                            distill_method=distill_method,
                            distill_space="encoded",
                            output_space="encoded",
                            convert_binary=False,
                            cluster_center=cluster_center,
                            encoder_name=en,
                            distill_size=N,
                            cache_dir=Path(PLOTS_CACHE_DIR) / "distilled_distribution",
                        ),
                        load_distilled_embeddings(
                            dataset_name=dataset_name,
                            reduce_method=reduce_method,
                            distill_method=distill_method,
                            distill_space="encoded",
                            output_space="decoded",
                            convert_binary=False,
                            cluster_center=cluster_center,
                            encoder_name=en,
                            distill_size=N,
                            cache_dir=Path(PLOTS_CACHE_DIR) / "distilled_distribution",
                        ),
                        load_distilled_embeddings(
                            dataset_name=dataset_name,
                            reduce_method=reduce_method,
                            distill_method=distill_method,
                            distill_space="encoded",
                            output_space="decoded",
                            convert_binary=True,
                            cluster_center=cluster_center,
                            encoder_name=en,
                            distill_size=N,
                            cache_dir=Path(PLOTS_CACHE_DIR) / "distilled_distribution",
                        ),
                    ]
                )
                for en in AUTOENCODERS + MULTIHEAD_AUTOENCODERS
            ]
        )
        fig = plot_distilled_distribution(
            distill_results,
            reduce_method=reduce_method,
            colormap=dataset_stats_colormap["labels"],
        )
        distilled_distribution[N] = json.loads(fig.to_json())
        progress.update(task, advance=1)
    json.dump(
        distilled_distribution,
        open(
            Path(PLOTS_DIR)
            / f"{dataset_name}_{distill_method}_{cluster_center}_{reduce_method}_distilled_distribution.json",
            "w",
        ),
    )

    if progress is not None and task is not None:
        progress.remove_task(task)


def make_feature_distribution_plot(
    dataset_name: str,
):
    dataset, labels = load_openml_dataset(
        Path(DATA_DIR) / dataset_name,
        DATASETS[dataset_name].download_url,
        DATASETS[dataset_name].label,
        raw_data=True,
    )

    fig = plot_feature_distribution(
        dataset=dataset,
        labels=labels,
        title=f"{dataset_name} Feature Statistics",
        colormap=dataset_stats_colormap["labels"],
    )
    fig.write_json(Path(PLOTS_DIR) / f"{dataset_name}_feature_distribution.json")


def make_dataset_stats_plot():
    stats = []
    with progress_bar() as progress:
        task = progress.add_task("Dataset Stats ", total=len(DATASETS))
        for ds in DATASETS:
            dset_stats = load_dataset_stats(
                dataset_name=ds,
                progress=progress,
            )
            if dset_stats is not None:
                stats += [dset_stats]
            progress.update(task, advance=1)
        stats = pd.concat(stats)

    fig = plot_dataset_stats(stats, dataset_stats_colormap)
    fig.write_json(Path(PLOTS_DIR) / "dataset_statistics.json")


def make_encoder_stats_plot():
    stats = []
    with progress_bar() as progress:
        task = progress.add_task("Encoder Stats ", total=len(DATASETS))
        for ds in DATASETS:
            enc_stats = load_enc_stats(
                dataset_name=ds,
                latent_dim=16,
                encoder_names=AUTOENCODERS,
            )
            if enc_stats is not None:
                stats += [enc_stats]
            progress.update(task, advance=1)
        stats = pd.concat(stats)

    fig = plot_autoencoder_stats(
        stats,
        encoder_stats_colormap,
    )
    fig.write_json(Path(PLOTS_DIR) / "encoder_statistics.json")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("plot_type")
    args = parser.parse_args()

    if args.plot_type == "classifier_overall":
        with progress_bar() as progress:
            overall_results_task = progress.add_task(
                "[green]Overall Results Plot ", total=len(CLASSIFIERS)
            )
            for clf in CLASSIFIERS:
                make_overall_results_plot(classifier_name=clf, tune_hyperopt=False)

                make_overall_results_plot(classifier_name=clf, tune_hyperopt=True)
                progress.update(overall_results_task, advance=1)

    elif args.plot_type == "classifier_results":
        with progress_bar() as progress:
            dataset_results_task = progress.add_task(
                "[green]Dataset Results Plot ", total=len(CLASSIFIERS) * len(DATASETS)
            )
            for ds in DATASETS:
                for clf in CLASSIFIERS:
                    make_distill_results_plot(
                        dataset_name=ds, classifier_name=clf, tune_hyperopt=False
                    )

                    make_distill_results_plot(
                        dataset_name=ds, classifier_name=clf, tune_hyperopt=True
                    )

                    make_baseline_results_plot(
                        dataset_name=ds, classifier_name=clf, tune_hyperopt=False
                    )

                    make_baseline_results_plot(
                        dataset_name=ds, classifier_name=clf, tune_hyperopt=True
                    )
                    progress.update(dataset_results_task, advance=1)

    elif args.plot_type == "classifier_params":
        with progress_bar() as progress:
            overall_task = progress.add_task(
                "[green]Classifier Stats Plots ", total=len(CLASSIFIERS) * len(DATASETS)
            )
            for clf in CLASSIFIERS:
                for ds in DATASETS:
                    make_tuned_parameters_plot(
                        dataset_name=ds,
                        classifier_name=clf,
                    )
                    progress.update(overall_task, advance=1)

    elif args.plot_type == "encoded_distribution":
        with progress_bar() as progress:
            overall_task = progress.add_task(
                "[green]Encoded Distribution Plots ",
                total=len(DATASETS) * len(REDUCE_METHODS) * len(AUTOENCODERS),
            )
            for ds in DATASETS:
                for rm in REDUCE_METHODS:
                    for en in AUTOENCODERS:
                        make_encoded_distribution_plot(
                            dataset_name=ds,
                            reduce_method=rm,
                            encoder_name=en,
                        )
                        progress.update(overall_task, advance=1)

    elif args.plot_type == "distilled_distribution":
        with progress_bar() as progress:
            overall_task = progress.add_task(
                "[green]Distilled Distribution Plots ",
                total=len(DATASETS) * len(REDUCE_METHODS) * len(AUTOENCODERS),
            )
            for ds in DATASETS:
                for dm in DISTILL_METHODS:
                    for cc in CLUSTER_CENTERS:
                        for rm in REDUCE_METHODS:
                            if dm == "kip" and cc == "closest":
                                continue
                            make_distilled_distribution_plot(
                                dataset_name=ds,
                                distill_method=dm,
                                cluster_center=cc,
                                reduce_method=rm,
                                progress=progress,
                            )
                            progress.update(overall_task, advance=1)
    elif args.plot_type == "feature_distribution":
        with progress_bar() as progress:
            overall_task = progress.add_task(
                "[green]Feature Distribution Plots ",
                total=len(DATASETS) * len(REDUCE_METHODS) * len(AUTOENCODERS),
            )
            for ds in DATASETS:
                make_feature_distribution_plot(ds)

    elif args.plot_type == "dataset_stats":
        make_dataset_stats_plot()

    elif args.plot_type == "encoder_stats":
        make_encoder_stats_plot()
