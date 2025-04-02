from pathlib import Path
from typing import Literal

import hydra
import pandas as pd
from omegaconf import DictConfig

from tabdd.config import (
    ClassifierTrainConfig,
    ClassifierTuneConfig,
    DataModeConfig,
    DatasetConfig,
    DistillConfig,
    EncoderTrainConfig,
    EncoderTuneConfig,
    MultiEncoderTuneConfig,
    load_classifier_train_config,
    load_classifier_tune_configs,
    load_data_mode_config,
    load_dataset_configs,
    load_distill_configs,
    load_encoder_train_config,
    load_encoder_tune_configs,
    load_pipeline_configs,
)
from tabdd.config.paths import RESULTS_CACHE_DIR, RUN_CONFIG_DIR
from tabdd.results.load import load_all_clf_perf
from tabdd.utils import progress_bar


def compute_clf_ranks(config: DictConfig) -> dict[str, pd.DataFrame]:
    data_mode_config = load_data_mode_config(config)
    dataset_configs = load_dataset_configs(config)
    encoder_tune_configs = load_encoder_tune_configs(config)
    encoder_train_config = load_encoder_train_config(config)
    distill_configs = load_distill_configs(config)
    classifier_tune_configs = load_classifier_tune_configs(config)
    classifier_train_config = load_classifier_train_config(config)

    save_dir = (
        Path(RESULTS_CACHE_DIR)
        / classifier_train_config.results_dir.name
        / classifier_train_config.results_dir.name
        / encoder_train_config.identifier
        / classifier_train_config.identifier
    )

    dm_save_path = save_dir / "rank_per_dm.csv"
    os_save_path = save_dir / "rank_per_os.csv"
    enc_save_path = save_dir / "rank_per_enc.csv"
    clf_save_path = save_dir / "rank_per_clf.csv"

    if (
        not dm_save_path.is_file()
        or not os_save_path.is_file()
        or not enc_save_path.is_file()
        or not clf_save_path.is_file()
        or config.recompute_rank
    ):
        results, incomplete = load_all_clf_perf(config)

        # filter out incomplete ones

        df = results[
            # ~results["Dataset"].isin(incomplete["Dataset"])
            # & 
            (results["Subset"] == "Test")
            & (
                (results["Output Space"] != "decoded")
                | ((results["Output Space"] == "decoded") & results["Convert Binary"])
            )
            & (
                (results["Data Mode"] != "Random Sample")
                # | ((results["Output Space"] == "decoded") & results["Convert Binary"])
            )
        ].copy()

        distill_methods = set(dm.pretty_name for dm in distill_configs) - set(
            ["Original", "Encoded", "Decoded", "Random Sample"]
        )

        base_encoders = set(enc.pretty_name for enc in encoder_tune_configs if isinstance(enc, EncoderTuneConfig))
        multi_encoders = set(enc.pretty_name for enc in encoder_tune_configs if isinstance(enc, MultiEncoderTuneConfig))

        rank_per_dm = []
        wlt_per_dm = []
        rank_per_enc = []
        rank_per_os = []
        rank_per_clf = []
        rank_per_enc_dm = []

        for (n, ds, clf), grouped in df.groupby(["N", "Dataset", "Classifier"]):
            for (enc, output_space, cc), by_dm in grouped[
                grouped["Distill Method"].isin(distill_methods)
            ].groupby(["Encoder", "Output Space", "Cluster Center"], dropna=False):
                rs = grouped[grouped["Distill Method"] == "Random Sample"]
                all_dms = pd.concat(
                    [
                        by_dm,
                        rs,
                    ]
                )

                if cc == "closest":
                    non_clusters = grouped[
                        (grouped["Distill Method"].isin(["KIP", "GM"]))
                        & (grouped["Encoder"] == enc)
                        & (grouped["Output Space"] == output_space)
                        # & (grouped["Convert Binary"]==cb)
                    ]
                    all_dms = pd.concat([all_dms, non_clusters])

                # take average
                score_per_dm = []
                for dm, perf_per_dm in all_dms.groupby("Distill Method"):
                    score_per_dm += [
                        {
                            "Distill Method": dm,
                            "Score": perf_per_dm["Score"].mean(),
                        }
                    ]
                score_per_dm = pd.DataFrame(score_per_dm)
                ranks = [
                    rank
                    for rank, sorted_idx in sorted(
                        enumerate(score_per_dm["Score"].values.argsort()[::-1]),
                        key=lambda x: x[1],
                    )
                ]
                wlt = [
                    (
                        int(
                            score
                            > score_per_dm[
                                score_per_dm["Distill Method"] == "Random Sample"
                            ]["Score"].values[0]
                        ),
                        int(
                            score
                            < score_per_dm[
                                score_per_dm["Distill Method"] == "Random Sample"
                            ]["Score"].values[0]
                        ),
                        int(
                            score
                            == score_per_dm[
                                score_per_dm["Distill Method"] == "Random Sample"
                            ]["Score"].values[0]
                        ),
                    )
                    for score in score_per_dm["Score"]
                ]

                rank_per_dm += [
                    {
                        "Distill Method": dm,
                        "Rank": rank,
                        "RS Win": win,
                        "RS Loss": loss,
                        "RS Tie": tie,
                        "N": n,
                        "Dataset": ds,
                        "Classifier": clf,
                        "Encoder": enc,
                        "Output Space": output_space,
                        # "Convert Binary": cb,
                        "Cluster Center": cc,
                    }
                    for dm, rank, (win, loss, tie) in zip(
                        score_per_dm["Distill Method"], ranks, wlt
                    )
                ]

            # Now for per encoder
            for (dm, output_space, cc), by_enc in grouped[
                ~grouped["Encoder"].isna()
            ].groupby(
                ["Distill Method", "Output Space", "Cluster Center"], dropna=False
            ):
                score_per_enc = pd.DataFrame(
                    [
                        {
                            "Encoder": enc,
                            "Score": perf_per_enc["Score"].mean(),
                        }
                        for enc, perf_per_enc in by_enc.groupby("Encoder")
                    ]
                )

                ranks = [
                    rank
                    for rank, sorted_idx in sorted(
                        enumerate(score_per_enc["Score"].values.argsort()[::-1]),
                        key=lambda x: x[1],
                    )
                ]
                # print(len(score_per_enc))

                rank_per_enc += [
                    {
                        "Encoder": enc,
                        "Distill Method": dm,
                        "Rank": rank,
                        "N": n,
                        "Dataset": ds,
                        "Classifier": clf,
                        "Output Space": output_space,
                        # "Convert Binary": cb,
                        "Cluster Center": cc,
                    }
                    for enc, rank in zip(score_per_enc["Encoder"], ranks)
                ]


            # Now for per output_space
            for (dm, cc), by_os in grouped.groupby(
                ["Distill Method", "Cluster Center"], dropna=False
            ):
                if dm.lower() not in distill_methods:
                    continue

                # print(dm, len(by_os))
                # print(by_os[["Output Space", "Convert Binary"]])
                #
                # quit()

                for base_enc in base_encoders:
                    by_os_enc = pd.concat(
                        [
                            by_os[by_os["Output Space"] == "original"],
                            by_os[
                                by_os["Encoder"].str.contains(base_enc, na=False)
                            ],
                        ]
                    )
                    score_per_os = pd.DataFrame(
                        [
                            {
                                "Output+Encoder Space": (
                                    "Original"
                                    if os == "original"
                                    else f"{os.capitalize()}"
                                    + ("-FT" if "multihead" in enc.lower() else "")
                                ),
                                "Output Space": os,
                                "Encoder": enc,
                                "Score": by_dm_os_enc["Score"].mean(),
                                "Regret": by_dm_os_enc["Regret"].mean(),
                            }
                            for (os, enc), by_dm_os_enc in by_os_enc.groupby(
                                ["Output Space", "Encoder"], dropna=False
                            )
                        ]
                    )

                    ranks = [
                        rank
                        for rank, sorted_idx in sorted(
                            enumerate(score_per_os["Score"].values.argsort()[::-1]),
                            key=lambda x: x[1],
                        )
                    ]
                    # print(len(score_per_enc))

                    rank_per_os += [
                        {
                            "Output+Encoder Space": os_enc,
                            "Output Space": os,
                            "Encoder": enc,
                            "Distill Method": dm,
                            "Rank": rank,
                            "N": n,
                            "Dataset": ds,
                            "Classifier": clf,
                            "Cluster Center": cc,
                            "Regret": reg,
                        }
                        for os, enc, os_enc, reg, rank, in zip(
                            score_per_os["Output Space"],
                            score_per_os["Encoder"],
                            score_per_os["Output+Encoder Space"],
                            score_per_os["Regret"],
                            ranks,
                        )
                    ]

        for (n, ds, dm, enc, os, cc), grouped in df.groupby(
            [
                "N",
                "Dataset",
                "Distill Method",
                "Encoder",
                "Output Space",
                "Cluster Center",
            ],
            dropna=False,
        ):
            regret_per_clf = pd.DataFrame(
                [
                    {"Classifier": clf, "Regret": perf_per_clf["Regret"].mean()}
                    for clf, perf_per_clf in grouped.groupby("Classifier")
                ]
            )
            ranks = [
                rank
                for rank, sorted_idx in sorted(
                    enumerate(regret_per_clf["Regret"].argsort()),
                    key=lambda x: x[1],
                )
            ]
            rank_per_clf += [
                {
                    "Output Space": os,
                    "Encoder": enc,
                    "Distill Method": dm,
                    "Rank": rank,
                    "N": n,
                    "Dataset": ds,
                    "Classifier": clf,
                    "Cluster Center": cc,
                }
                for clf, rank in zip(regret_per_clf["Classifier"], ranks)
            ]



        rank_per_dm = pd.DataFrame(rank_per_dm)
        rank_per_dm.to_csv(dm_save_path, index=False)

        rank_per_enc = pd.DataFrame(rank_per_enc)
        rank_per_enc.to_csv(enc_save_path, index=False)

        rank_per_os = pd.DataFrame(rank_per_os)
        rank_per_os.to_csv(os_save_path, index=False)

        rank_per_clf = pd.DataFrame(rank_per_clf)
        rank_per_clf.to_csv(clf_save_path, index=False)
    else:
        rank_per_dm = pd.read_csv(dm_save_path)
        rank_per_enc = pd.read_csv(enc_save_path)
        rank_per_os = pd.read_csv(os_save_path)
        rank_per_clf = pd.read_csv(clf_save_path)

    return {
        "dm": rank_per_dm,
        "os": rank_per_os,
        "enc": rank_per_enc,
        "clf": rank_per_clf,
    }


@hydra.main(version_base=None, config_path=RUN_CONFIG_DIR, config_name="all")
def run(config: DictConfig) -> None:
    compute_clf_ranks(config)


if __name__ == "__main__":
    run()
