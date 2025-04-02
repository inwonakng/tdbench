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


cmap = colormaps["tab20c"]
METHOD_GROUPS = {
    "KIP TF": {
        "filter": partial(
            encoded_distill, distill_method="KIP", encoder="TF-MultiHead"
        ),
        "color": "green",
        "linestyle": "-",
    },
    "KIP FFN": {
        "filter": partial(
            encoded_distill, distill_method="KIP", encoder="MLP-MultiHead"
        ),
        "color": "green",
        "linestyle": "--",
    },
    "KIP GNN": {
        "filter": partial(
            encoded_distill, distill_method="KIP", encoder="GNN-MultiHead"
        ),
        "color": "green",
        "linestyle": "-.",
    },
    "GM TF": {
        "filter": partial(encoded_distill, distill_method="GM", encoder="TF-MultiHead"),
        "color": "brown",
        "linestyle": "-",
    },
    "GM FFN": {
        "filter": partial(
            encoded_distill, distill_method="GM", encoder="MLP-MultiHead"
        ),
        "color": "brown",
        "linestyle": "--",
    },
    "GM GNN": {
        "filter": partial(
            encoded_distill, distill_method="GM", encoder="GNN-MultiHead"
        ),
        "color": "brown",
        "linestyle": "-.",
    },
    "AG TF": {
        "filter": partial(
            encoded_distill, distill_method="Agglo", encoder="TF-MultiHead"
        ),
        "color": "red",
        "linestyle": "-",
    },
    "AG FFN": {
        "filter": partial(
            encoded_distill, distill_method="Agglo", encoder="MLP-MultiHead"
        ),
        "color": "red",
        "linestyle": "--",
    },
    "AG GNN": {
        "filter": partial(
            encoded_distill, distill_method="Agglo", encoder="GNN-MultiHead"
        ),
        "color": "red",
        "linestyle": "-.",
    },
    "KM TF": {
        "filter": partial(
            encoded_distill, distill_method="KMeans", encoder="TF-MultiHead"
        ),
        "color": "blue",
        "linestyle": "-",
    },
    "KM FFN": {
        "filter": partial(
            encoded_distill, distill_method="KMeans", encoder="MLP-MultiHead"
        ),
        "color": "blue",
        "linestyle": "--",
    },
    "KM GNN": {
        "filter": partial(
            encoded_distill, distill_method="KMeans", encoder="GNN-MultiHead"
        ),
        "color": "blue",
        "linestyle": "-.",
    },
    # "GM": {
    #     "filter": partial(any_distill, distill_method="GM"),
    #     "color": "brown",
    #     "linestyle": "-",
    # },
    # "AG": {
    #     "filter": partial(any_distill, distill_method="Agglo"),
    #     "color": "red",
    #     "linestyle": "-",
    # },
    # "KM": {
    #     "filter": partial(any_distill, distill_method="KMeans"),
    #     "color": "blue",
    #     "linestyle": "-",
    # },
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


"""
Processing complete
"""

"""
Table of comparison through all N
"""

distill_methods = [
    "KM",
    "AG",
    "KIP",
    "GM",
]
dm_colors = ["blue", "red", "green", "brown"]
encoders = [" TF", " FFN", " GNN"]

for n in [10, 50, 100]:
    filtered = finalized2[finalized2["N"] == n]
    rnd_scores = filtered[filtered["Method"].isin(["RND"])]["Regret"]
    table = [
        {
            ("Distill Method", ""): "Random Sample",
            ("Encoder", ""): "N/A",
            ("Regret", "Min"): f"{rnd_scores.min():.4f}",
            ("Regret", "Q1"): f"{rnd_scores.quantile(0.25):.4f}",
            ("Regret", "Mean"): f"{rnd_scores.mean():.4f}",
            ("Regret", "Median"): f"{rnd_scores.median():.4f}",
            ("Regret", "Q3"): f"{rnd_scores.quantile(0.75):.4f}",
            ("Regret", "Max"): f"{rnd_scores.max():.4f}",
        }
    ]
    for dm in distill_methods:
        for enc in encoders:
            scores = filtered[filtered["Method"].isin([dm + enc])]["Regret"]
            table.append(
                {
                    ("Distill Method", ""): dm,
                    ("Encoder", ""): enc,
                    ("Regret", "Min"): f"{scores.min():.4f}",
                    ("Regret", "Q1"): f"{scores.quantile(0.25):.4f}",
                    ("Regret", "Mean"): f"{scores.mean():.4f}",
                    ("Regret", "Median"): f"{scores.median():.4f}",
                    ("Regret", "Q3"): f"{scores.quantile(0.75):.4f}",
                    ("Regret", "Max"): f"{scores.max():.4f}",
                }
            )
    table = pd.DataFrame(table)
    table.columns = pd.MultiIndex.from_tuples(table.columns)
    print(table.to_markdown(index=False))
    with open(f"iclr-figures/rq2-compare-distill-method-n{n}.tex", "w") as f:
        table_str = table.to_latex(index=False)
        bef, aft = table_str.split("\\midrule\n")
        mid, aft = aft.split("\\bottomrule\n")
        rows = mid.split("\\\\\n")[:-1]
        new_rows = []
        prev_dm = ""
        for i, r in enumerate(rows):
            cur_dm = r.split("&")[0].strip()
            if i > 0 and i < len(rows) - 1 and prev_dm != cur_dm:
                new_rows.append("\\midrule")
            new_rows.append(r + "\\\\")
            prev_dm = cur_dm
        new_table = bef + "\\midrule\n" + "\n".join(new_rows) + "\n\\bottomrule\n" + aft
        new_table = new_table.replace("{llllllll}", "{llcccccc}").replace(
            "\multicolumn{6}{r}", "\multicolumn{6}{c}"
        )
        f.write(new_table)

"""
Crit Diff. plot of N = 10
"""

for n in [10, 50, 100]:
    filtered = finalized2[finalized2["N"] == n]
    rnd_scores = filtered[filtered["Method"].isin(["RND"])]["Regret"]
    sco_per_dm = pd.DataFrame(
        {
            dm: filtered[filtered["Method"].isin([dm + enc for enc in encoders])][
                "Regret"
            ]
            for dm in distill_methods
        }
    )
    sco_per_dm["RND"] = rnd_scores.tolist() * len(encoders)
    rank_per_dm = sco_per_dm.rank(axis=1)
    ww = sp.posthoc_wilcoxon(
        sco_per_dm.melt(),
        val_col="value",
        group_col="variable",
    )
    fig, ax = plt.subplots(figsize=(3, 1.5))
    sp.critical_difference_diagram(
        ranks=rank_per_dm.mean(),
        sig_matrix=ww,
        ax=ax,
        color_palette={**dict(zip(distill_methods, dm_colors)), "RND": "black"},
    )
    fig.tight_layout()
    fig.savefig(
        f"iclr-figures/rq2-distill-method-crit-diff-n{n}.pdf", bbox_inches="tight"
    )
