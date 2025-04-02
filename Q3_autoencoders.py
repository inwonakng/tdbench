import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import colormaps
from functools import partial
import scikit_posthocs as sp


# df = pd.read_csv("./parsed_with_rel_reg.csv")


df = pd.read_csv("./data_mode_switch_results.csv", low_memory=False)
# df_mix_tf = pd.read_csv("./mixed_tf_results.csv")
# df = pd.concat([df, df_mix_tf])
df = df[df["Subset"] == "Test"]
df["Use Encoder"] = df["Encoder"].notna()
df = df[df["Classifier"] != "GaussianNB"]

enc_stats = pd.read_csv("./enc_stats.csv")
enc_stats["Model"] = enc_stats["Model"].replace(
    {"MLP": "FFN", "MLP-MultiHead": "FFN-MultiHead"}
)


def mixed_distill(df, distill_method):
    return df[
        (df["Distill Method"] == distill_method)
        & (df["Data Parse Mode"] == "mixed")
        & (df["Post Data Parse Mode"] == "mixed")
        & (~df["Use Encoder"])
    ]


# def any_distill(df, distill_method):
#     return df[
#         (df["Distill Method"] == distill_method)
#         & (df["Data Parse Mode"] == "onehot")
#         & (df["Use Encoder"])
#     ]


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
METHOD_GROUPS = {
    "TF* KIP": {
        "filter": partial(
            encoded_distill, distill_method="KIP", encoder="TF-MultiHead"
        ),
        "color": "green",
        "linestyle": "-",
    },
    "FFN* KIP": {
        "filter": partial(
            encoded_distill, distill_method="KIP", encoder="MLP-MultiHead"
        ),
        "color": "green",
        "linestyle": "--",
    },
    "GNN* KIP": {
        "filter": partial(
            encoded_distill, distill_method="KIP", encoder="GNN-MultiHead"
        ),
        "color": "green",
        "linestyle": "-.",
    },
    "TF* GM": {
        "filter": partial(encoded_distill, distill_method="GM", encoder="TF-MultiHead"),
        "color": "brown",
        "linestyle": "-",
    },
    "FFN* GM": {
        "filter": partial(
            encoded_distill, distill_method="GM", encoder="MLP-MultiHead"
        ),
        "color": "brown",
        "linestyle": "--",
    },
    "GNN* GM": {
        "filter": partial(
            encoded_distill, distill_method="GM", encoder="GNN-MultiHead"
        ),
        "color": "brown",
        "linestyle": "-.",
    },
    "TF* AG": {
        "filter": partial(
            encoded_distill, distill_method="Agglo", encoder="TF-MultiHead"
        ),
        "color": "red",
        "linestyle": "-",
    },
    "FFN* AG": {
        "filter": partial(
            encoded_distill, distill_method="Agglo", encoder="MLP-MultiHead"
        ),
        "color": "red",
        "linestyle": "--",
    },
    "GNN* AG": {
        "filter": partial(
            encoded_distill, distill_method="Agglo", encoder="GNN-MultiHead"
        ),
        "color": "red",
        "linestyle": "-.",
    },
    "TF* KM": {
        "filter": partial(
            encoded_distill, distill_method="KMeans", encoder="TF-MultiHead"
        ),
        "color": "blue",
        "linestyle": "-",
    },
    "FFN* KM": {
        "filter": partial(
            encoded_distill, distill_method="KMeans", encoder="MLP-MultiHead"
        ),
        "color": "blue",
        "linestyle": "--",
    },
    "GNN* KM": {
        "filter": partial(
            encoded_distill, distill_method="KMeans", encoder="GNN-MultiHead"
        ),
        "color": "blue",
        "linestyle": "-.",
    },
    "TF KIP": {
        "filter": partial(encoded_distill, distill_method="KIP", encoder="TF"),
        "color": "green",
        "linestyle": "-",
    },
    "FFN KIP": {
        "filter": partial(encoded_distill, distill_method="KIP", encoder="MLP"),
        "color": "green",
        "linestyle": "--",
    },
    "GNN KIP": {
        "filter": partial(encoded_distill, distill_method="KIP", encoder="GNN"),
        "color": "green",
        "linestyle": "-.",
    },
    "TF GM": {
        "filter": partial(encoded_distill, distill_method="GM", encoder="TF"),
        "color": "brown",
        "linestyle": "-",
    },
    "FFN GM": {
        "filter": partial(encoded_distill, distill_method="GM", encoder="MLP"),
        "color": "brown",
        "linestyle": "--",
    },
    "GNN GM": {
        "filter": partial(encoded_distill, distill_method="GM", encoder="GNN"),
        "color": "brown",
        "linestyle": "-.",
    },
    "TF AG": {
        "filter": partial(encoded_distill, distill_method="Agglo", encoder="TF"),
        "color": "red",
        "linestyle": "-",
    },
    "FFN AG": {
        "filter": partial(encoded_distill, distill_method="Agglo", encoder="MLP"),
        "color": "red",
        "linestyle": "--",
    },
    "GNN AG": {
        "filter": partial(encoded_distill, distill_method="Agglo", encoder="GNN"),
        "color": "red",
        "linestyle": "-.",
    },
    "TF KM": {
        "filter": partial(encoded_distill, distill_method="KMeans", encoder="TF"),
        "color": "blue",
        "linestyle": "-",
    },
    "FFN KM": {
        "filter": partial(encoded_distill, distill_method="KMeans", encoder="MLP"),
        "color": "blue",
        "linestyle": "--",
    },
    "GNN KM": {
        "filter": partial(encoded_distill, distill_method="KMeans", encoder="GNN"),
        "color": "blue",
        "linestyle": "-.",
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


"""
Processing complete
"""

"""
Table of comparison through all N
"""

distill_methods = [
    " KM",
    " AG",
    " KIP",
    " GM",
]
encoders = [
    "TF",
    "FFN",
    "GNN",
    "TF*",
    "FFN*",
    "GNN*",
]

"""
Crit Diff. plot of N = 10
"""

sco_per_enc = pd.DataFrame(
    {
        enc: finalized2[
            finalized2["Method"].isin([enc + dm for dm in distill_methods])
        ]["Regret"]
        for enc in encoders
    }
)
rank_per_enc = sco_per_enc.rank(axis=1)


avg_rank = rank_per_enc.mean().to_dict()
med_sco = sco_per_enc.median().to_dict()

with open("iclr-figures/rq3-enc-rank.tex", "w") as f:
    table = pd.DataFrame(
        [
            {
                "Encoder": k,
                "Mean Rank": f"{rank:.4f}",
                "Median Regret": f"{med_sco[k]:.4f}",
            }
            for k, rank in avg_rank.items()
        ]
    )
    f.write(table.to_latex(index=False))

for n in [10, 50, 100]:
    filtered = finalized2[finalized2["N"] == n]
    rnd_scores = filtered[filtered["Method"].isin(["RND"])]["Regret"]
    sco_per_enc = pd.DataFrame(
        {
            enc: filtered[
                filtered["Method"].isin([enc + dm for dm in distill_methods])
            ]["Regret"]
            for enc in encoders
        }
    )
    rank_per_enc = sco_per_enc.rank(axis=1)
    ww = sp.posthoc_wilcoxon(
        sco_per_enc.melt(),
        val_col="value",
        group_col="variable",
    )
    fig, ax = plt.subplots(figsize=(4, 3))
    sp.critical_difference_diagram(
        ranks=rank_per_enc.mean(),
        sig_matrix=ww,
        ax=ax,
    )
    fig.tight_layout()
    fig.savefig(f"iclr-figures/rq3-enc-crit-diff-n{n}.pdf", bbox_inches="tight")
    avg_rank = rank_per_enc.mean().to_dict()
    med_sco = sco_per_enc.median().to_dict()
    with open(f"iclr-figures/rq3-enc-rank-n{n}.tex", "w") as f:
        table = pd.DataFrame(
            [
                {
                    "Encoder": k,
                    "Mean Rank": f"{rank:.4f}",
                    "Median Regret": f"{med_sco[k]:.4f}",
                }
                for k, rank in avg_rank.items()
            ]
        )
        f.write(table.to_latex(index=False))


filtered = finalized2[finalized2["N"] == 10]

colors = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
]

plot_data = []
for encoder in ["TF", "GNN", "FFN"]:
    by_enc = filtered[filtered["Method"].str.contains(encoder)]
    for dataset, by_enc_ds in by_enc.groupby("Dataset"):
        param_size = enc_stats[
            (enc_stats["Model"].str.contains(encoder))
            & (enc_stats["Dataset"] == dataset)
        ]["Encoder Params"].values[0]
        reg = by_enc_ds["Regret"].mean()
        plot_data.append(
            {
                "Dataset": dataset,
                "Encoder": encoder,
                "Regret": reg,
                "Param Size": param_size,
            }
        )
plot_data = pd.DataFrame(plot_data)

fig, ax = plt.subplots(figsize=(4, 2.5))
for color, (encoder, group) in zip(colors, plot_data.groupby("Encoder")):
    x_med = np.median(group["Regret"])
    y_med = np.median(group["Param Size"])
    ax.scatter(x_med, y_med, color=color)
    x_left, x_right = np.quantile(group["Regret"], [0.25, 0.75])
    y_left, y_right = np.quantile(group["Param Size"], [0.25, 0.75])
    ax.errorbar(
        x_med,
        y_med,
        xerr=[
            [x_med - x_left],
            [x_right - x_med],
        ],
        yerr=[
            [y_med - y_left],
            [y_right - y_med],
        ],
    )
    if encoder in ["TF", "MLP"]:
        text_loc = (
            x_med - 0.05,
            y_med + 10000,
        )
    else:
        text_loc = (
            x_med + 0.01,
            y_med + 10000,
        )
    ax.text(
        *text_loc,
        encoder,
        color=color,
        bbox=dict(
            facecolor="#ededed",
            edgecolor="black",
            boxstyle="round",
            pad=0.2,
        ),
    )
ax.ticklabel_format(axis='y',style='sci',scilimits=(0,0))
ax.set_xlabel("Relative Regret")
ax.set_ylabel("# Params")
# ax.set_yscale("log")
fig.savefig("./iclr-figures/rq3-enc-param-vs-reg.pdf", bbox_inches="tight")
