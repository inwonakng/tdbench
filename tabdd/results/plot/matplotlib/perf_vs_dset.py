import hydra
from omegaconf import DictConfig
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

from tabdd.results.load import load_all_dataset_stats, load_all_clf_perf, compute_clf_ranks
from tabdd.config.paths import FIGURES_DIR, RUN_CONFIG_DIR
from tabdd.config import (
    load_data_mode_config,
    load_dataset_configs,
)

# hydra.initialize(
#     config_path="./config",
#     version_base=None,
# )
# config = hydra.compose(
#     config_name="all",
#     overrides=["refresh_results=false"]
# )

@hydra.main(version_base=None, config_path=RUN_CONFIG_DIR, config_name="all")
def run(config: DictConfig) -> None:

    perf, incomplete = load_all_clf_perf(config)
    ds_stats = load_all_dataset_stats(config)
    ranks = compute_clf_ranks(config)

    test_perf = perf[
        (perf["Subset"]=="Test")
        & (perf["N"] == 50)
        & ~perf["Distill Method"].isin(["Decoded","Original","Encoded"])
        & ~perf["Dataset"].isin(incomplete["Dataset"])
    ]

    ds_stats["Continuous Ratio"] = ds_stats["Continuous Features"] / (ds_stats["Categorical Features"] + ds_stats["Continuous Features"])

    perf_and_dset = test_perf.merge(ds_stats, on="Dataset")

    ranks_and_dset = {
        k: v.merge(ds_stats, on="Dataset")
        for k,v in ranks.items()
    }

    avg_scores = []
    for (dataset, clf, dm), subset in perf_and_dset.groupby(["Dataset","Classifier","Data Mode"]):
        score, regret = subset[["Score", "Regret"]].mean()

        avg_scores.append({
            "Dataset": dataset,
            "Continuous Ratio": subset["Continuous Ratio"].values[0],
            "Rows": subset["Rows"].values[0],
            "Classifier": clf,
            "Encoder": subset["Encoder"].values[0],
            "Data Mode": dm,
            "Score": score,
            "Regret": regret,
        })

    avg_scores = pd.DataFrame(avg_scores)


    datasets_by_ratio = []
    for cr, rows in ds_stats.groupby("Continuous Ratio"):
        dsets = rows["Dataset"].unique().tolist()
        datasets_by_ratio.append("\n".join([f"{cr:.4f}"]+dsets[:3])+("\n..." if len(dsets)>2 else ""))

    datasets_by_row = []
    for rc, rows in ds_stats.groupby("Rows"):
        dsets = rows["Dataset"].unique().tolist()
        datasets_by_row.append("\n".join([f"{rc:,d}"]+dsets[:3])+("\n..." if len(dsets)>2 else ""))

    comp_pretty_name = {
        "dm": "Distill Method",
        "os": "Output Space",
        "enc": "Encoder",
        "clf": "Classifier",
    }

    for comp, with_dset in ranks_and_dset.items():
        fig, ax = plt.subplots(figsize=(10,4))

        comp_name = comp_pretty_name[comp]

        p = sns.pointplot(
            x = "Continuous Ratio",
            y = "Rank",
            ax = ax,
            data = with_dset,
            hue = comp_name,
        )
        p.set(
            title = f"{comp_name} Rank by Dataset Feature Ratio",
        )
        p.set_xticklabels(
            labels=datasets_by_ratio,
            rotation=30
        )
        fig.tight_layout()

        fig.savefig(FIGURES_DIR/f"{comp}_rank_by_fratio.pdf",bbox_inches="tight")

        fig, ax = plt.subplots(figsize=(16,4))

        p = sns.pointplot(
            x = "Rows",
            y = "Rank",
            ax = ax,
            data = with_dset,
            hue = comp_name,
        )
        p.set(
            title = f"{comp_name} Rank by Dataset Row Count",
        )
        p.set_xticklabels(
            labels=datasets_by_row,
            rotation=30
        )
        fig.tight_layout()

        fig.savefig(FIGURES_DIR/f"{comp}_rank_by_rows.pdf",bbox_inches="tight")

if __name__ == "__main__":
    run()
