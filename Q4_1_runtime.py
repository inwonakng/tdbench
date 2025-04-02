import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from functools import partial
import scikit_posthocs as sp
from matplotlib import colormaps
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from pathlib import Path


# df = pd.read_csv("./parsed_with_rel_reg.csv")


df = pd.read_csv("./data_mode_switch_results.csv", low_memory=False)
# df_mix_tf = pd.read_csv("./mixed_tf_results.csv")
# df = pd.concat([df, df_mix_tf])
df = df[df["Subset"] == "Test"]
df["Use Encoder"] = df["Encoder"].notna()
df = df[df["Classifier"] != "GaussianNB"]
df["Encoder"] = df["Encoder"].replace(
    {
        "MLP": "FFN",
        "TF-MultiHead": "TF*",
        "GNN-MultiHead": "GNN*",
        "MLP-MultiHead": "FFN*",
    }
)
df["Distill Method"] = df["Distill Method"].replace(
    {
        "KMeans": "KM",
        "Agglo": "AG",
    }
)


# TODO: get encoder train times and add to the runtime figure.

enc_conversion = {
    "tfautoencoder": "TF",
    "multiheadtfautoencoder": "TF*",
    "gnnautoencoder": "GNN",
    "multiheadgnnautoencoder": "GNN*",
    "mlpautoencoder": "FFN",
    "multiheadmlpautoencoder": "FFN*",
}

enc_runtimes = []
for f in Path("./best_checkpoints/").glob(
    "*/*/onehot/standard/uniform-10/asha_hyperopt/16/"
):
    most_recent = sorted(f.glob("*/metrics.csv"))[-1]
    enc = enc_conversion[most_recent.parts[1]]
    dset = most_recent.parts[2]
    metrics = pd.read_csv(most_recent)
    if "*" in enc:
        best_idx = metrics["val/combined_score"].argmax()
    else:
        best_idx = metrics["val/recon_accuracy_score"].argmax()
    runtime_till_best = metrics.iloc[: best_idx + 1]["time_this_iter_s"].sum()
    runtime = metrics["time_this_iter_s"].sum()
    enc_runtimes.append(
        {
            "Encoder": enc,
            "Dataset": dset,
            "Runtime": runtime,
            "Runtime Till Best": runtime,
        }
    )
enc_runtimes = pd.DataFrame(enc_runtimes)


print(len(df["Dataset"].unique()), len(enc_runtimes["Dataset"].unique()))

df["Data Distill Time"].loc[df["Distill Method"] == "Original"] = 0
# df["Data Distill Time"][df["Data Distill Time"] == -1] = np.nan
# df["Default Train Time"][df["Default Train Time"] == -1] = np.nan
# df["Inference Time"][df["Inference Time"] == -1] = np.nan


def mixed_distill(df, distill_method):
    return df[
        (df["Distill Method"] == distill_method)
        & (df["Data Parse Mode"] == "mixed")
        & (df["Post Data Parse Mode"] == "mixed")
        & (~df["Use Encoder"])
    ]


def encoded_distill(df, distill_method, encoder):
    return df[
        (df["Distill Method"] == distill_method)
        & (df["Data Parse Mode"] == "onehot")
        & (df["Encoder"] == encoder)
        & (df["Use Encoder"])
    ]


def no_encode_distill(df, distill_method):
    return df[(df["Distill Method"] == distill_method) & (~df["Use Encoder"])]


cmap = colormaps["tab20c"]

encoders = [
    ("TF", "p"),
    ("GNN", "+"),
    ("FFN", "x"),
    ("TF*", "*"),
    ("GNN*", "P"),
    ("FFN*", "X"),
]

distill_methods = [
    ("KM", "blue"),
    ("AG", "red"),
    ("KIP", "green"),
    ("GM", "brown"),
]

METHOD_GROUPS = {
    "ORI": {
        "filter": partial(mixed_distill, distill_method="Original"),
        "color": "slategray",
        "linestyle": "o",
    },
    "RND": {
        "filter": partial(mixed_distill, distill_method="Random Sample"),
        "color": "black",
        "linestyle": "o",
    },
}
for dm, dm_c in distill_methods:
    METHOD_GROUPS[dm] = {
        "filter": partial(mixed_distill, distill_method=dm),
        "color": dm_c,
        "linestyle": "o",
    }
    for enc, enc_m in encoders:
        METHOD_GROUPS[f"{dm} {enc} Enc."] = {
            "filter": partial(encoded_distill, distill_method=dm, encoder=enc),
            "color": dm_c,
            "linestyle": enc_m,
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
        data = []
        enc_runtime = 0
        if "Enc." in k:
            enc = filtered["Encoder"].unique()[0]
            enc_runtime = enc_runtimes[
                (enc_runtimes["Dataset"] == dset) & (enc_runtimes["Encoder"] == enc)
            ]["Runtime"].values[0]
            if "*" in enc:
                enc_runtime += enc_runtimes[
                    (enc_runtimes["Dataset"] == dset)
                    & (enc_runtimes["Encoder"] == enc.replace("*", ""))
                ]["Runtime"].values[0]
        for (n, sn), by_n_sn in filtered.groupby(["N", "Short Name"]):
            good_filter = (by_n_sn["Data Distill Time"] != -1) & (
                by_n_sn["Default Train Time"] != -1
            )
            by_n_sn = by_n_sn[good_filter]
            if by_n_sn.empty:
                continue
            runtime = (
                by_n_sn["Data Distill Time"].mean()
                + by_n_sn["Default Train Time"].mean()
                + enc_runtime
            )
            data.append(
                (n, (ori_sco - by_n_sn["Score"].mean()) / (ori_sco - rnd_sco), runtime)
            )
        if len(data) == 0:
            print(f"{clf}:{dset} -- {k} Skipped.. No data")
            continue
        best_scos = pd.DataFrame(
            [
                (n, by_n["Regret"].min(), by_n["Runtime"].min())
                for n, by_n in pd.DataFrame(
                    data, columns=["N", "Regret", "Runtime"]
                ).groupby("N")
            ],
            columns=["N", "Regret", "Runtime"],
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


"""
Processing complete
"""
# TODO: Add CI (hori/vert lines)

fig, ax = plt.subplots(figsize=(4, 4))
for k, grp in finalized2.groupby("Method"):
    x_med = grp["Runtime"].median()
    y_med = grp["Regret"].median()
    x_left, x_right = np.quantile(grp["Runtime"], [0.25, 0.75])
    y_bottom, y_top = np.quantile(grp["Regret"], [0.25, 0.75])
    ax.scatter(
        x_med,
        y_med,
        label=k,
        color=METHOD_GROUPS[k]["color"],
        marker=METHOD_GROUPS[k]["linestyle"],
        edgecolor="black",
        zorder=10,
    )
    ax.errorbar(
        x_med,
        y_med,
        xerr=[
            [x_med - x_left],
            [x_right - x_med],
        ],
        yerr=[
            [y_med - y_bottom],
            [y_top - y_med],
        ],
        color=METHOD_GROUPS[k]["color"],
        alpha=0.5,
        zorder=1,
        elinewidth=1.5,
    )
ax.set_xlabel("Runtime (s)")
ax.set_ylabel("Relative Regret")
color_patches = [
    Patch(
        label="Original",
        facecolor="slategray",
        edgecolor="slategray",
    ),
    Patch(
        label="RND",
        facecolor="black",
        edgecolor="black",
    ),
] + [
    Patch(
        label=dm,
        facecolor=dm_c,
        edgecolor=dm_c,
    )
    for dm, dm_c in distill_methods
]
markers = [
    Line2D(
        [0],
        [0],
        marker="o",
        color="black",
        label="Original",
        markerfacecolor="black",
        linestyle="",
        markersize=7,
    )
] + [
    Line2D(
        [0],
        [0],
        marker=enc_m,
        color="black",
        label=enc,
        markerfacecolor="black",
        linestyle="",
        markersize=7,
    )
    for enc, enc_m in encoders
]
legend_c = ax.legend(
    handles=color_patches,
    loc="upper left",
    title="Distill Methods",
    bbox_to_anchor=(1, 0.47),
)
legend_m = ax.legend(
    handles=markers,
    loc="lower left",
    title="Encoders",
    bbox_to_anchor=(1, 0.47),
)
ax.add_artist(legend_c)
ax.add_artist(legend_m)
fig.tight_layout()
fig.savefig(
    "./iclr-figures/rq4_1-reg-vs-runtime.pdf",
    bbox_inches="tight",
    bbox_extra_artists=(
        legend_c,
        legend_m,
    ),
)
