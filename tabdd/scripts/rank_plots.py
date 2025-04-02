import pandas as pd

import matplotlib.pyplot as plt
from tabdd.results.plot.matplotlib import (
    make_boxplot,
    get_cmap,
)

colormap = [get_cmap("tab10")(i) for i in range(10)]

rank_per_dm = pd.read_csv("./rank_per_dm.csv")
rank_per_dm = rank_per_dm[rank_per_dm["Cluster Center"] == "centroid"]
rank_per_enc = pd.read_csv("./rank_per_enc.csv")
rank_per_enc = rank_per_enc[rank_per_enc["Cluster Center"] == "centroid"]
rank_per_os = pd.read_csv("./rank_per_os.csv")
rank_per_os = rank_per_os[rank_per_os["Cluster Center"] == "centroid"]
rank_per_clf = pd.read_csv("./rank_per_clf.csv")
rank_per_clf = rank_per_clf[rank_per_clf["Cluster Center"] == "centroid"]

CLASSIFIERS = rank_per_clf["Classifier"].unique()
ENCODERS = rank_per_enc["Encoder"].unique()
DISTILL_METHODS = list(
    set(rank_per_dm["Distill Method"].unique()) - set(["Random Sample"])
)


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


import numpy as np

all_distill_methods = ["Random Sample", "Agglo", "KIP", "KMeans"]


def get_vs_matrix(ranks):
    vs_matrix = np.zeros((4, 4))
    for _, one_comparison in ranks.groupby(
        ["N", "Dataset", "Classifier", "Encoder", "Output Space", "Cluster Center"]
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


fig, axes = plt.subplots(
    ncols=len(ENCODERS),
    nrows=3,
    figsize=(2 * len(ENCODERS), 4),
    sharey=True,
    sharex=True,
)

for i, N in enumerate([10, 50, 100]):
    for j, enc in enumerate(ENCODERS):
        axes[i, j].set_title(f"n/L={N}, {short_enc_name(enc)}")
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
fig.savefig("figures/rank_per_dm_enc_N.pdf", bbox_inches="tight")

fig, axes = plt.subplots(
    ncols=len(CLASSIFIERS),
    nrows=len(ENCODERS),
    figsize=(2 * len(CLASSIFIERS), 2 * len(ENCODERS)),
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
fig.savefig("figures/rank_per_dm_enc_clf.pdf", bbox_inches="tight")


fig, axes = plt.subplots(
    nrows=3,
    ncols=len(DISTILL_METHODS),
    figsize=(2 * len(DISTILL_METHODS), 4),
    sharey=True,
    sharex=True,
)

for i, N in enumerate([10, 50, 100]):
    for j, dm in enumerate(DISTILL_METHODS):
        axes[i, j].set_title(f"n/L={N}, {short_dm_name(dm)}")
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
fig.savefig("figures/rank_per_enc_dm_N.pdf", bbox_inches="tight")


fig, axes = plt.subplots(
    nrows=3,
    ncols=len(DISTILL_METHODS) + 1,
    figsize=(2 * (len(DISTILL_METHODS) + 1), 4),
    sharey=True,
    sharex=True,
)

for i, N in enumerate([10, 50, 100]):
    for j, dm in enumerate(["Random Sample"] + DISTILL_METHODS):
        axes[i, j].set_title(f"n/L={N}, {short_dm_name(dm)}")
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
            show_text=False
            # text_offsets=[0.1]*4,
        )
# fig.tight_layout()
fig.subplots_adjust(wspace=0.05, hspace=0.4)
fig.savefig("figures/rank_per_clf_dm_N.pdf", bbox_inches="tight")

fig, axes = plt.subplots(
    nrows=3,
    ncols=len(DISTILL_METHODS),
    figsize=(2 * len(DISTILL_METHODS), 4),
    sharey=True,
    sharex=True,
)

for i, N in enumerate([10, 50, 100]):
    for j, dm in enumerate(DISTILL_METHODS):
        axes[i, j].set_title(f"n/L={N}, {short_dm_name(dm)}")
        x = []
        y = []
        for os, each_os in rank_per_os[
            (rank_per_os["N"] == N) & (rank_per_os["Distill Method"] == dm)
        ].groupby("Output+Encoder Space"):
            x += [os.replace("Encoded", "Latent").replace("DecodedBinary", "Decoded")]
            y += [each_os["Rank"].tolist()]

        make_boxplot(
            x,
            y,
            ax=axes[i, j],
            colormap=colormap,
            show_text=False,
        )

    if i == 2:
        for ax in axes[i, :]:
            ax.tick_params(
                axis="x",
                rotation=45,
            )
fig.subplots_adjust(wspace=0.05, hspace=0.4)
fig.savefig("figures/rank_per_os_dm_N.pdf", bbox_inches="tight")

# logreg and original space
fig, axes = plt.subplots(
    nrows=3,
    ncols=len(ENCODERS),
    figsize=(2 * len(ENCODERS), 4),
    sharey=True,
    sharex=True,
)

for i, N in enumerate([10, 50, 100]):
    for j, enc in enumerate(ENCODERS):
        axes[i, j].set_title(f"n/L={N}, {short_enc_name(enc)}")
        x = []
        y = []
        for dm, each_dm in rank_per_dm[
            (rank_per_dm["N"] == N)
            & (rank_per_dm["Encoder"] == enc)
            & (rank_per_dm["Classifier"] == "LogisticRegression")
        ].groupby("Distill Method"):
            x += [short_dm_name(dm)]
            y += [each_dm["Rank"].tolist()]

        make_boxplot(x, y, ax=axes[i, j], colormap=colormap, show_text=False)

fig.subplots_adjust(wspace=0.05, hspace=0.4)
fig.savefig("figures/rank_per_dm_enc_N_logreg.pdf", bbox_inches="tight")


"""
VS MATRIX
"""

all_vs_matrix = get_vs_matrix(rank_per_dm)
xgb_vs_matrix = get_vs_matrix(
    rank_per_dm[(rank_per_dm["Classifier"] == "XGBClassifier")],
)
mlp_vs_matrix = get_vs_matrix(
    rank_per_dm[(rank_per_dm["Classifier"] == "MLPClassifier")],
)


import seaborn as sns

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
fig.savefig("figures/dm_vs_matrices.pdf", bbox_inches="tight")
