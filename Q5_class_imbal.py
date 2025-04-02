import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import colormaps
from functools import partial
import scikit_posthocs as sp
from matplotlib.lines import Line2D


# df = pd.read_csv("./parsed_with_rel_reg.csv")


df = pd.read_csv("./data_mode_switch_results.csv", low_memory=False)
# df_mix_tf = pd.read_csv("./mixed_tf_results.csv")
# df = pd.concat([df, df_mix_tf])
df = df[df["Subset"] == "Test"]
df["Use Encoder"] = df["Encoder"].notna()
df = df[~df["Classifier"].isin(["GaussianNB"])]

ds_stats = pd.read_csv("./dataset_stats.csv")
ds_stats = ds_stats[ds_stats["Dataset"].isin(df["Dataset"])]


def mixed_distill(df, distill_method):
    return df[
        (df["Distill Method"] == distill_method)
        & (df["Data Parse Mode"] == "mixed")
        & (df["Post Data Parse Mode"] == "mixed")
        & (~df["Use Encoder"])
    ]


def onehot_distill_enc(df, distill_method):
    return df[
        (df["Distill Method"] == distill_method)
        & (df["Data Parse Mode"] == "onehot")
        & (df["Post Data Parse Mode"] == "onehot")
        & (df["Output Space"] == "encoded")
        & (df["Use Encoder"])
    ]


def onehot_distill_mixed(df, distill_method):
    return df[
        (df["Distill Method"] == distill_method)
        & (df["Data Parse Mode"] == "onehot")
        & (df["Post Data Parse Mode"] == "mixed")
        & (df["Output Space"].isin(["decoded", "original"]))
        & (df["Use Encoder"])
    ]


cmap = colormaps["tab20c"]
METHOD_GROUPS = {
    "KIP": {
        "filter": partial(onehot_distill_enc, distill_method="KIP"),
        "color": "green",
        "linestyle": "-",
    },
    "GM": {
        "filter": partial(onehot_distill_enc, distill_method="GM"),
        "color": "brown",
        "linestyle": "-",
    },
    "AG": {
        "filter": partial(onehot_distill_enc, distill_method="Agglo"),
        "color": "red",
        "linestyle": "-",
    },
    "KM": {
        "filter": partial(onehot_distill_enc, distill_method="KMeans"),
        "color": "blue",
        "linestyle": "-",
    },
    "RND": {
        "filter": partial(mixed_distill, distill_method="Random Sample"),
        "color": "black",
        "linestyle": "-",
    },
}


to_cat = []
for (dset, clf), by_dset_clf in df.groupby(
    [
        "Dataset",
        "Classifier",
    ]
):
    ori_sco = by_dset_clf[
        (by_dset_clf["Distill Method"] == "Original")
        & (by_dset_clf["Data Parse Mode"] == "mixed")
        & (by_dset_clf["Post Data Parse Mode"] == "mixed")
    ]["Score"].mean()
    rnd_sco = by_dset_clf[
        (by_dset_clf["Distill Method"] == "Random Sample")
        & (by_dset_clf["Data Parse Mode"] == "mixed")
        & (by_dset_clf["Post Data Parse Mode"] == "mixed")
        & (by_dset_clf["N"] == 10)
    ]["Score"].mean()
    if rnd_sco >= ori_sco:
        print(f"{clf}:{dset} Skipped.. Too easy")
        continue
    for k, grp in METHOD_GROUPS.items():
        filtered = grp["filter"](by_dset_clf)
        if filtered.empty:
            print(f"{clf}:{dset} -- {k} Skipped.. No data")
            continue
        data = [
            (n, (ori_sco - by_n_sn["Score"].mean()) / (ori_sco - rnd_sco))
            for (n, sn), by_n_sn in filtered.groupby(["N", "Short Name"])
        ]
        if len(data) == 0:
            print(f"{clf}:{dset} -- {k} Skipped.. No data")
            continue
        best_scos = pd.DataFrame(
            [
                (n, by_n["Regret"].min())
                for n, by_n in pd.DataFrame(data, columns=["N", "Regret"]).groupby("N")
            ],
            columns=["N", "Regret"],
        )
        best_scos["Method"] = k
        best_scos["Dataset"] = dset
        best_scos["Classifier"] = clf
        to_cat.append(best_scos)
finalized = pd.concat(to_cat)

clf_comp = {}
for clf, by_clf in finalized.groupby("Classifier"):
    min_dsets = set(by_clf["Dataset"].unique())
    fair_comp = {}
    for k, grp in METHOD_GROUPS.items():
        scores = by_clf[by_clf["Method"] == k]
        min_dsets &= set(scores["Dataset"].unique())
        fair_comp[k] = scores
    print("-" * 40)
    print(f"{clf} with {len(min_dsets)} datasets")
    print("=" * 40 + "\n")
    fair_comp = {k: v[v["Dataset"].isin(min_dsets)] for k, v in fair_comp.items()}
    clf_comp[clf] = fair_comp
finalized2 = pd.concat([vv for v in clf_comp.values() for vv in v.values()])


cmp_lab = finalized2.copy()
cmp_lab = finalized2[~finalized2["Classifier"].isin(["FTTransformer", "ResNet"])].copy()
n = 10
cmp_lab_n = cmp_lab[cmp_lab["N"] == n]


"""
Processing complete
"""


fig, axes = plt.subplots(
    ncols=2,
    nrows=2,
    figsize=(4, 3),
    layout="constrained",
    sharex=True,
    sharey=True,
)
for i in range(4):
    ax = axes[i // 2, i % 2]
    k = "RND"
    grp = cmp_lab_n[cmp_lab_n["Method"] == k]
    regs = grp.groupby(["Dataset"])["Regret"].median().reset_index()
    regs = regs.merge(ds_stats, on="Dataset", how="left")
    regs = regs.sort_values("Minority Label Ratio")
    x = regs["Minority Label Ratio"].values
    y = regs["Regret"].values
    # m, b = np.polyfit(x, y, deg=1)
    # ax.scatter(
    #     x,
    #     y,
    #     label=k,
    #     color=METHOD_GROUPS[k]["color"],
    #     s=5,
    #     alpha=0.7,
    # )
    ax.axline(
        (0, 1),
        slope=0,
        color=METHOD_GROUPS[k]["color"],
        linestyle=METHOD_GROUPS[k]["linestyle"],
        linewidth=1,
    )
for i, (k, grp) in enumerate(cmp_lab_n[cmp_lab_n["Method"] != "RND"].groupby("Method")):
    ax = axes[i // 2, i % 2]
    regs = []
    for dset, dgrp in grp.groupby("Dataset"):
        regs.append({
            "Regret": dgrp["Regret"].median(),
            "Dataset": dset,
            "Minority Label Ratio": ds_stats[ds_stats["Dataset"] == dset]["Minority Label Ratio"].values[0],
            "iqr_25": dgrp["Regret"].quantile(0.25),
            "iqr_75": dgrp["Regret"].quantile(0.75),
            "iqr_range": dgrp["Regret"].quantile(0.75) - dgrp["Regret"].quantile(0.25),
        })
    regs = pd.DataFrame(regs)
    x = regs["Minority Label Ratio"].values
    y = regs["Regret"].values
    s = regs["iqr_range"] / regs["iqr_range"].max()
    m, b = np.polyfit(x, y, deg=1)
    ax.scatter(
        x,
        y,
        label=k,
        color=METHOD_GROUPS[k]["color"],
        # s=20 * s,
        s = 5,
        alpha=0.7,
    )
    ax.axline(
        (0, b),
        slope=m,
        color=METHOD_GROUPS[k]["color"],
        linestyle=METHOD_GROUPS[k]["linestyle"],
        linewidth=1,
    )
    ax.set_title(k)
fig.supxlabel("Minority Label Ratio")
fig.supylabel("Regret")
lastax = axes[0, -1]
markers = [
    Line2D(
        [0, 0.5],
        [0, 0],
        # marker=".",
        color="black",
        label="RND",
        # markerfacecolor="black",
        linestyle="-",
        # markersize=7,
    )
]
legend_m = lastax.legend(
    handles=markers,
    loc="lower right",
    bbox_to_anchor=(1, -0.15),
)
lastax.add_artist(legend_m)
fig.savefig(
    "iclr-figures/rq5-class-imbal.pdf",
    # bbox_inches="tight",
    bbox_extra_artists=[legend_m],
)
