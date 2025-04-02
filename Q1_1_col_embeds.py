import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import colormaps
from functools import partial
from tqdm.auto import tqdm


# df = pd.read_csv("./parsed_with_rel_reg.csv")


df = pd.read_csv("./data_mode_switch_results.csv", low_memory=False)
df_mix_tf = pd.read_csv("./mixed_tf_results.csv")
df_ple_tf = pd.read_csv("./ple_tf_results.csv")
df = pd.concat([df, df_mix_tf, df_ple_tf])
df = df[df["Subset"] == "Test"]
df["Use Encoder"] = df["Encoder"].notna()
df = df[~df["Classifier"].isin(["GaussianNB", "FTTransformer", "ResNet"])]
df = df[df["Dataset"] != "PhishingWebsites"]


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
        # & (df["Output Space"] == "encoded")
        & (df["Use Encoder"])
    ]


def mixed_distill_enc(df, distill_method):
    return df[
        (df["Distill Method"] == distill_method)
        & (df["Data Parse Mode"] == "mixed")
        & (df["Post Data Parse Mode"] == "mixed")
        # & (df["Output Space"] == "encoded")
        & (df["Use Encoder"])
    ]


def ple_distill_enc(df, distill_method):
    return df[
        (df["Distill Method"] == distill_method)
        & (df["Data Parse Mode"] == "ple")
        & (df["Post Data Parse Mode"] == "ple")
        # & (df["Output Space"] == "encoded")
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


mixed_distill(
    df=df[df["Classifier"] == "FTTransformer"], distill_method="Random Sample"
)["Dataset"].nunique()

onehot_distill_enc(df=df[df["Classifier"] == "FTTransformer"], distill_method="KMeans")[
    "Dataset"
].nunique()

cmap = colormaps["tab20c"]
METHOD_GROUPS = {
    "KIP Bin Enc.": {
        "filter": partial(onehot_distill_enc, distill_method="KIP"),
        "color": "green",
        "linestyle": "-",
    },
    "KIP Scaled Enc.": {
        "filter": partial(mixed_distill_enc, distill_method="KIP"),
        "color": "green",
        "linestyle": "--",
    },
    "KIP PLE Enc.": {
        "filter": partial(ple_distill_enc, distill_method="KIP"),
        "color": "green",
        "linestyle": "-.",
    },
    "GM Bin Enc.": {
        "filter": partial(onehot_distill_enc, distill_method="GM"),
        "color": "brown",
        "linestyle": "-",
    },
    "GM Scaled Enc.": {
        "filter": partial(mixed_distill_enc, distill_method="GM"),
        "color": "brown",
        "linestyle": "--",
    },
    "GM PLE Enc.": {
        "filter": partial(ple_distill_enc, distill_method="GM"),
        "color": "brown",
        "linestyle": "-.",
    },
    "AG Bin Enc.": {
        "filter": partial(onehot_distill_enc, distill_method="Agglo"),
        "color": "red",
        "linestyle": "-",
    },
    "AG Scaled Enc.": {
        "filter": partial(mixed_distill_enc, distill_method="Agglo"),
        "color": "red",
        "linestyle": "--",
    },
    "AG PLE Enc.": {
        "filter": partial(ple_distill_enc, distill_method="Agglo"),
        "color": "red",
        "linestyle": "-.",
    },
    "KM Bin Enc.": {
        "filter": partial(onehot_distill_enc, distill_method="KMeans"),
        "color": "blue",
        "linestyle": "-",
    },
    "KM Scaled Enc.": {
        "filter": partial(mixed_distill_enc, distill_method="KMeans"),
        "color": "blue",
        "linestyle": "--",
    },
    "KM PLE Enc.": {
        "filter": partial(ple_distill_enc, distill_method="KMeans"),
        "color": "blue",
        "linestyle": "-.",
    },
    "RND": {
        "filter": partial(mixed_distill, distill_method="Random Sample"),
        "color": "black",
        "linestyle": "-",
    },
}

to_cat = []
for (dset, clf), by_dset_clf in df.groupby(["Dataset", "Classifier"]):
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
finalized = pd.concat([vv for v in clf_comp.values() for vv in v.values()])

# fig, axes = plt.subplots(
#     nrows=1,
#     ncols=len(clf_comp),
#     sharex=True,
#     sharey=True,
#     figsize=(12, 3),
# )
# for i, (clf, fair_comp) in enumerate(clf_comp.items()):
#     ax = axes[i]
#     for k, v in fair_comp.items():
#         ax.plot(
#             v.groupby("N")["Regret"].median(),
#             label=k,
#             c=METHOD_GROUPS[k]["color"],
#             ls=METHOD_GROUPS[k]["linestyle"],
#         )
#         # individual lines with alpha
#         for _, per_dset in v.groupby("Dataset"):
#             ax.plot(
#                 per_dset.groupby("N")["Regret"].mean(),
#                 alpha=0.05,
#                 c=METHOD_GROUPS[k]["color"],
#                 ls=METHOD_GROUPS[k]["linestyle"],
#             )
#     ax.set_title(clf)
#     ax.set_ylim(-1, 2)
#     ax.grid()
# axes[-1].legend(ncol=2, fontsize=8, bbox_to_anchor=(1.02, 0.7))
# fig.savefig("iclr-figures/compare-col-emb-median.pdf", bbox_inches="tight")
#
#
# fig, axes = plt.subplots(
#     nrows=1,
#     ncols=len(clf_comp),
#     sharex=True,
#     sharey=True,
#     figsize=(12, 3),
# )
# for i, (clf, fair_comp) in enumerate(clf_comp.items()):
#     ax = axes[i]
#     for k, v in fair_comp.items():
#         ax.plot(
#             v.groupby("N")["Regret"].median(),
#             label=k,
#             c=METHOD_GROUPS[k]["color"],
#             ls=METHOD_GROUPS[k]["linestyle"],
#         )
#         ax.fill_between(
#             v["N"].unique(),
#             v.groupby("N")["Regret"].quantile(0.25).values,
#             v.groupby("N")["Regret"].quantile(0.75).values,
#             color=METHOD_GROUPS[k]["color"],
#             alpha=0.05,
#         )
#     ax.set_title(clf)
#     ax.set_ylim(-1, 2)
#     ax.grid()
# axes[-1].legend(ncol=2, fontsize=8, bbox_to_anchor=(1.02, 0.7))
# fig.tight_layout()
# fig.savefig("iclr-figures/compare-col-emb-quantile.pdf", bbox_inches="tight")


"""
Custom plot for comparing the data modes of distill methods
"""

fig, axes = plt.subplots(
    nrows=len(clf_comp),
    ncols=4,
    sharex=True,
    sharey="row",
    figsize=(12, 8),
)
for i, (clf, fair_comp) in enumerate(clf_comp.items()):
    for j, dm in enumerate(["KIP", "GM", "AG", "KM"]):
        ax = axes[i, j]
        for prefix in [" Bin Enc.", " Scaled Enc.", " PLE Enc."]:
            k = dm + prefix
            v = fair_comp[k]
            ax.plot(
                v.groupby("N")["Regret"].median(),
                label=k,
                c=METHOD_GROUPS[k]["color"],
                ls=METHOD_GROUPS[k]["linestyle"],
            )
            ax.fill_between(
                v["N"].unique(),
                v.groupby("N")["Regret"].quantile(0.25).values,
                v.groupby("N")["Regret"].quantile(0.75).values,
                color=METHOD_GROUPS[k]["color"],
                alpha=0.05,
            )
        k = "RND"
        v = fair_comp[k]
        ax.plot(
            v.groupby("N")["Regret"].median(),
            label=k,
            c=METHOD_GROUPS[k]["color"],
            ls=METHOD_GROUPS[k]["linestyle"],
        )
        ax.fill_between(
            v["N"].unique(),
            v.groupby("N")["Regret"].quantile(0.25).values,
            v.groupby("N")["Regret"].quantile(0.75).values,
            color=METHOD_GROUPS[k]["color"],
            alpha=0.05,
        )
        ax.set_title(f"{clf} - {dm}")
        # ax.set_ylim(-3, 2)
        ax.grid()
        if i == 0:
            axes[i, j].legend(ncol=2, fontsize=8)
fig.tight_layout()
fig.savefig("iclr-figures/compare-col-emb-per-dm.pdf", bbox_inches="tight")


"""
RQ1.1: Which column embedding scheme leads to more useful latent space?
"""

print("RQ1.1: Which column embedding scheme leads to more useful latent space?")
distill_methods = ["KM", "AG", "GM", "KIP"]
for n in [10, 50, 100]:
    table = []
    for postfix in [" Bin Enc.", " Scaled Enc.", " PLE Enc."]:
        row = {}
        for dm in distill_methods:
            q1 = finalized[
                    (finalized["N"] == n) & (finalized["Method"] == dm + postfix)
                ]["Regret"].quantile(0.25)
            med = finalized[
                    (finalized["N"] == n) & (finalized["Method"] == dm + postfix)
                ]["Regret"].median()
            q3 = finalized[
                    (finalized["N"] == n) & (finalized["Method"] == dm + postfix)
                ]["Regret"].quantile(0.75)
            row[dm] = f"$_{{{q1:.4f}}}$ {med:.4f} $_{{{q3:.4f}}}$"
            # row[(dm, "Q1")] = "{:.4f}".format()
            # row[(dm, "Med.")] = "{:.4f}".format()
            # row[(dm, "Q3")] = "{:.4f}".format()
        table.append(row)
    table = pd.DataFrame(table)
    # table.columns = pd.MultiIndex.from_tuples(table.columns)
    # table["RND"] = "{:.4f} {:.4f} {:.4f}".format(
    #     finalized[(finalized["N"] == n) & (finalized["Method"] == "RND")][
    #         "Regret"
    #     ].quantile(0.25),
    #     finalized[(finalized["N"] == n) & (finalized["Method"] == "RND")][
    #         "Regret"
    #     ].median(),
    #     finalized[(finalized["N"] == n) & (finalized["Method"] == "RND")][
    #         "Regret"
    #     ].quantile(0.75),
    # )
    # othercols = table.columns
    table["Col. Emb."] = ["Binary", "Scaled", "PLE"]
    table = table[["Col. Emb."] + distill_methods]
    print(f"N = {n}\n")
    print(table.to_markdown(floatfmt=".4f", index=False))
    print()
    with open(f"iclr-figures/rq1_1-compare-col-emb-n{n}.tex", "w") as f:
        f.write(table.to_latex(index=False).replace("{lllll}", "{lcccc}"))
