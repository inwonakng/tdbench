import hydra
from omegaconf import DictConfig
import pandas as pd
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from tabdd.config.paths import RESULTS_CACHE_DIR, RUN_CONFIG_DIR, FIGURES_DIR
from tabdd.results.load import compute_clf_ranks
from tabdd.results.plot.matplotlib import (
    make_boxplot,
    get_cmap,
)

colormap = [get_cmap("tab10")(i) for i in range(10)]

CLASSIFIERS = [
    "XGBClassifier",
    "LogisticRegression",
    "MLPClassifier",
    "KNeighborsClassifier",
    "GaussianNB",
]
ENCODERS = [
    "MLP",
    "MLP-MultiHead",
    "GNN",
    "GNN-MultiHead",
    "TF",
    "TF-MultiHead",
]
DISTILL_METHODS = [
    "KMeans",
    "Agglo",
    "KIP",
]


def short_dm_name(dm):
    if dm == "Random Sample":
        return "RS"
    elif dm == "KMeans":
        return "KM"
    elif dm == "Agglo":
        return "AG"
    else:
        return dm


def short_clf_name(clf):
    if clf == "XGBClassifier":
        return "XGB"
    elif clf == "MLPClassifier":
        return "MLP"
    elif clf == "LogisticRegression":
        return "LR"
    elif clf == "GaussianNB":
        return "NB"
    elif clf == "KNeighborsClassifier":
        return "KNN"
    else:
        return clf


def short_enc_name(enc):
    return enc.replace("MLP", "FFN").replace("MultiHead", "FT")


all_distill_methods = ["Random Sample", "Agglo", "KIP", "KMeans"]


def get_vs_matrix(ranks):
    vs_matrix = np.zeros((4, 4))
    for _, one_comparison in ranks.groupby(
        [
            "N",
            "Dataset",
            "Classifier",
            "Encoder",
            "Output Space",
            # "Convert Binary",
            "Cluster Center",
        ]
    ):
        for i, dm1 in enumerate(all_distill_methods):
            for j, dm2 in enumerate(all_distill_methods):
                if dm1 == dm2:
                    continue
                rank1 = one_comparison[one_comparison["Distill Method"] == dm1][
                    "Rank"
                ].values[0]
                rank2 = one_comparison[one_comparison["Distill Method"] == dm2][
                    "Rank"
                ].values[0]

                if rank1 < rank2:
                    vs_matrix[i, j] += 1
    total_instances = len(
        list(
            ranks.groupby(
                [
                    "N",
                    "Dataset",
                    "Classifier",
                    "Encoder",
                    "Output Space",
                    # "Convert Binary",
                    "Cluster Center",
                ]
            )
        )
    )
    vs_matrix = pd.DataFrame(
        vs_matrix / total_instances,
        columns=[short_dm_name(dm) for dm in all_distill_methods],
        index=[short_dm_name(dm) for dm in all_distill_methods],
    )

    return vs_matrix


def dm_enc_N(rank_per_dm):
    fig, axes = plt.subplots(
        nrows=3, ncols=len(ENCODERS), figsize=(8, 4), sharey=True, sharex=True
    )

    for i, N in enumerate([10, 50, 100]):
        for j, enc in enumerate(ENCODERS):
            axes[i, j].set_title(f"N={N}, {short_enc_name(enc)}")
            x = []
            y = []
            for dm, each_dm in rank_per_dm[
                (rank_per_dm["N"] == N) & (rank_per_dm["Encoder"] == enc)
            ].groupby("Distill Method"):
                x += [short_dm_name(dm)]
                y += [each_dm["Rank"].tolist()]

            make_boxplot(
                x,
                y,
                ax=axes[i, j],
                colormap=colormap,
                show_text=False,
                # text_offsets=[0.1]*4,
            )
    # fig.suptitle("Rank per DM by encoder and N")
    fig.subplots_adjust(wspace=0.05, hspace=0.4)
    # fig.tight_layout()
    # fig.savefig("figures/rank_per_dm_enc_N.pdf", bbox_inches="tight")
    return fig


def dm_enc_clf(rank_per_dm):

    # fig, axes = plt.subplots(nrows=4, ncols=5, figsize=(12,12), sharey=True, sharex=True)
    fig, axes = plt.subplots(
        nrows=len(ENCODERS),
        ncols=len(CLASSIFIERS),
        figsize=(8, 6),
        sharey=True,
        sharex=True,
    )

    for i, enc in enumerate(ENCODERS):
        for j, clf in enumerate(CLASSIFIERS):
            axes[i, j].set_title(f"{short_enc_name(enc)}, {short_clf_name(clf)}")
            x = []
            y = []
            for dm, each_dm in rank_per_dm[
                (rank_per_dm["Classifier"] == clf) & (rank_per_dm["Encoder"] == enc)
            ].groupby("Distill Method"):
                x += [short_dm_name(dm)]
                y += [each_dm["Rank"].tolist()]

            make_boxplot(
                x,
                y,
                ax=axes[i, j],
                colormap=colormap,
                # text_offsets=[0.1]*4,
                show_text=False,
            )
    # fig.suptitle("Rank per DM by classifier and N")
    # fig.tight_layout()
    fig.subplots_adjust(wspace=0.05, hspace=0.4)
    # fig.savefig("figures/rank_per_dm_enc_clf.pdf", bbox_inches="tight")
    return fig


def dm_vs_matrics(rank_per_dm):
    all_vs_matrix = get_vs_matrix(rank_per_dm)
    xgb_vs_matrix = get_vs_matrix(
        rank_per_dm[rank_per_dm["Classifier"] == "XGBClassifier"],
    )
    mlp_vs_matrix = get_vs_matrix(
        rank_per_dm[rank_per_dm["Classifier"] == "MLPClassifier"],
    )
    fig, axes = plt.subplots(figsize=(9, 3.5), ncols=3)
    sns.heatmap(
        all_vs_matrix,
        ax=axes[0],
        annot=True,
        cbar=False,
        cmap="Blues",
    )
    axes[0].set_title("Overall")

    sns.heatmap(
        xgb_vs_matrix,
        ax=axes[1],
        annot=True,
        cbar=False,
        cmap="Blues",
    )
    axes[1].set_title("XGBoost")

    sns.heatmap(
        mlp_vs_matrix,
        ax=axes[2],
        annot=True,
        cbar=False,
        cmap="Blues",
    )
    axes[2].set_title("MLP")

    fig.tight_layout()
    # fig.savefig("figures/dm_vs_matrices.pdf", bbox_inches="tight")
    return fig


def enc_dm_N(rank_per_enc):
    fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(7, 4), sharey=True, sharex=True)

    for i, N in enumerate([10, 50, 100]):
        for j, dm in enumerate(DISTILL_METHODS):
            axes[i, j].set_title(f"N={N}, {dm}")
            x = []
            y = []
            for enc, each_enc in rank_per_enc[
                (rank_per_enc["N"] == N) & (rank_per_enc["Distill Method"] == dm)
            ].groupby("Encoder"):
                x += [short_enc_name(enc)]
                y += [each_enc["Rank"].tolist()]
            make_boxplot(
                x,
                y,
                ax=axes[i, j],
                colormap=colormap,
                show_text=False,
                # rotate_xlabels=(None if i != 2 else 45),
                # text_offsets=[0.1]*4,
            )
        if i == 2:
            for ax in axes[i, :]:
                ax.tick_params(
                    axis="x",
                    rotation=30,
                )
    # fig.suptitle("Rank per Encoder by DM and N")
    # fig.tight_layout()
    #
    # for ax in axes[-1, :]:
    #     ax.set_xticklabels(x, rotation=45, ha='center')
    fig.subplots_adjust(wspace=0.05, hspace=0.4)
    # fig.savefig("figures/rank_per_enc_dm_N.pdf", bbox_inches="tight")
    return fig


def clf_dm_N(rank_per_clf):
    fig, axes = plt.subplots(nrows=3, ncols=4, figsize=(8, 4), sharey=True, sharex=True)

    for i, N in enumerate([10, 50, 100]):
        for j, dm in enumerate(["Random Sample", "KMeans", "Agglo", "KIP"]):
            axes[i, j].set_title(f"N={N}, {short_dm_name(dm)}")
            x = []
            y = []
            for clf, each_clf in rank_per_clf[
                (rank_per_clf["N"] == N) & (rank_per_clf["Distill Method"] == dm)
            ].groupby("Classifier"):
                x += [short_clf_name(clf)]
                y += [each_clf["Rank"].tolist()]

            make_boxplot(
                x,
                y,
                ax=axes[i, j],
                colormap=colormap,
                show_text=False,
                # text_offsets=[0.1]*4,
            )
    # fig.tight_layout()
    fig.subplots_adjust(wspace=0.05, hspace=0.4)
    # fig.savefig("figures/rank_per_clf_dm_N.pdf", bbox_inches="tight")
    return fig


@hydra.main(version_base=None, config_path=RUN_CONFIG_DIR, config_name="all")
def run(config: DictConfig):
    ranks = compute_clf_ranks(config)

    # fig = dm_enc_N(ranks["dm"])
    # fig.savefig(FIGURES_DIR / "rank_per_dm_enc_N_new.pdf", bbox_inches="tight")

    # fig = dm_enc_clf(ranks["dm"])
    # fig.savefig(FIGURES_DIR / "rank_per_dm_enc_clf_new.pdf", bbox_inches="tight")

    # fig = dm_vs_matrics(ranks["dm"])
    # fig.savefig(FIGURES_DIR / "dm_vs_matrices_new.pdf", bbox_inches="tight")

    fig = enc_dm_N(ranks["enc"])
    fig.savefig(FIGURES_DIR / "rank_per_enc_dm_N_new.pdf", bbox_inches="tight")

    fig = clf_dm_N(ranks["clf"])
    fig.savefig(FIGURES_DIR / "rank_per_clf_dm_N_new.pdf", bbox_inches="tight")

if __name__ == "__main__":
    run()
