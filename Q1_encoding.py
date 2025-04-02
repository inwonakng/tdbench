import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import colormaps
from functools import partial
from tqdm.auto import tqdm
import numpy as np

# df = pd.read_csv("./parsed_with_rel_reg.csv")


df = pd.read_csv("./data_mode_switch_results.csv", low_memory=False)
# df_mix_tf = pd.read_csv("./mixed_tf_results.csv")
# df = pd.concat([df, df_mix_tf])
df = df[df["Subset"] == "Test"]
df["Use Encoder"] = df["Encoder"].notna()
df = df[df["Classifier"] != "GaussianNB"]
df["Classifier"] = df["Classifier"].map(
    {
        "XGBClassifier": "XGB",
        "KNeighborsClassifier": "KNN",
        "LogisticRegression": "LR",
        "MLPClassifier": "MLP",
        "FTTransformer": "FTT",
        "ResNet": "RN",
    }
)


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
        "filter": partial(mixed_distill, distill_method="KIP"),
        "color": "green",
        "linestyle": "-",
    },
    "KIP Enc.": {
        "filter": partial(onehot_distill_enc, distill_method="KIP"),
        "color": "green",
        "linestyle": "--",
    },
    "KIP Rec.": {
        "filter": partial(onehot_distill_mixed, distill_method="KIP"),
        "color": "green",
        "linestyle": "-.",
    },
    "GM": {
        "filter": partial(mixed_distill, distill_method="GM"),
        "color": "brown",
        "linestyle": "-",
    },
    "GM Enc.": {
        "filter": partial(onehot_distill_enc, distill_method="GM"),
        "color": "brown",
        "linestyle": "--",
    },
    "GM Rec.": {
        "filter": partial(onehot_distill_mixed, distill_method="GM"),
        "color": "brown",
        "linestyle": "-.",
    },
    "AG": {
        "filter": partial(mixed_distill, distill_method="Agglo"),
        "color": "red",
        "linestyle": "-",
    },
    "AG Enc.": {
        "filter": partial(onehot_distill_enc, distill_method="Agglo"),
        "color": "red",
        "linestyle": "--",
    },
    "AG Rec.": {
        "filter": partial(onehot_distill_mixed, distill_method="Agglo"),
        "color": "red",
        "linestyle": "-.",
    },
    "KM": {
        "filter": partial(mixed_distill, distill_method="KMeans"),
        "color": "blue",
        "linestyle": "-",
    },
    "KM Enc.": {
        "filter": partial(onehot_distill_enc, distill_method="KMeans"),
        "color": "blue",
        "linestyle": "--",
    },
    "KM Rec.": {
        "filter": partial(onehot_distill_mixed, distill_method="KMeans"),
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
finalized2 = pd.concat([vv for v in clf_comp.values() for vv in v.values()])

"""
Processing complete
"""

boxprops = dict(
    boxstyle="round",
    facecolor="white",
    alpha=0.8,
    ec=None,
)
fig, axes = plt.subplots(
    nrows=2,
    ncols=2,
    # sharex=True,
    # sharey=True,
    figsize=(5, 4),
    layout="constrained",
)
for i, clf in enumerate(["XGB", "KNN", "LR", "MLP"]):
    fair_comp = clf_comp[clf]
    ax = axes.ravel()[i]
    plot_data = {}
    for k, v in fair_comp.items():
        if k not in ["RND", "KM Enc.", "KM"]:
            continue
        medians = v.groupby("N")["Regret"].median()
        plot_data[k] = medians
        ax.plot(
            medians,
            label=k,
            c=METHOD_GROUPS[k]["color"],
            ls=METHOD_GROUPS[k]["linestyle"],
        )
    km_ori = plot_data["KM"].values[0]
    km_enc = plot_data["KM Enc."].values[0]
    ax.arrow(
        5,
        km_ori,
        0,
        km_enc - km_ori,
        color="red",
        width=0.5,
        head_length=0.05,
        head_width=2,
        length_includes_head=True,
    )
    improvement = (km_ori - km_enc) / km_ori * 100
    ax.annotate(
        f"{improvement:.2f}%",
        (5, plot_data["KM Enc."].values[0]),
        xycoords="data",
        xytext=(20, 15),
        textcoords="offset points",
        # textcoords="axes fraction",
        bbox=boxprops,
        # arrowprops=arrowprops,
        horizontalalignment="center",
        verticalalignment="top",
        fontsize=8,
    )
    ax.set_title(clf)
    # ax.set_ylim(-0.1, 1.1)
    ax.grid()
fig.supxlabel("Instances Per Class")
fig.supylabel("Relative Regret")
axes[0, -1].legend(ncol=1, fontsize=8, bbox_to_anchor=(1.02, 0.7))
fig.savefig("iclr-figures/intro-figure.pdf", bbox_inches="tight")


fig, axes = plt.subplots(
    nrows=1,
    ncols=len(clf_comp),
    sharex=True,
    sharey=True,
    figsize=(16, 3),
)
for i, (clf, fair_comp) in enumerate(clf_comp.items()):
    ax = axes[i]
    for k, v in fair_comp.items():
        ax.plot(
            v.groupby("N")["Regret"].median(),
            label=k,
            c=METHOD_GROUPS[k]["color"],
            ls=METHOD_GROUPS[k]["linestyle"],
        )
        # individual lines with alpha
        for _, per_dset in v.groupby("Dataset"):
            ax.plot(
                per_dset.groupby("N")["Regret"].mean(),
                alpha=0.05,
                c=METHOD_GROUPS[k]["color"],
                ls=METHOD_GROUPS[k]["linestyle"],
            )
    ax.set_title(clf)
    ax.set_ylim(-1, 2)
    ax.grid()
axes[-1].legend(ncol=2, fontsize=8, bbox_to_anchor=(1.02, 0.7))
fig.savefig("iclr-figures/rq1-dm-per-clf-median.pdf", bbox_inches="tight")


fig, axes = plt.subplots(
    nrows=1,
    ncols=len(clf_comp),
    sharex=True,
    sharey=True,
    figsize=(16, 3),
)
for i, (clf, fair_comp) in enumerate(clf_comp.items()):
    ax = axes[i]
    for k, v in fair_comp.items():
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
    ax.set_title(clf)
    ax.set_ylim(-1, 2)
    ax.grid()
axes[-1].legend(ncol=2, fontsize=8, bbox_to_anchor=(1.02, 0.7))
fig.tight_layout()
fig.savefig("iclr-figures/rq1-dm-per-clf-quantile.pdf", bbox_inches="tight")


fig, axes = plt.subplots(
    nrows=len(clf_comp),
    ncols=len(list(clf_comp.values())[0]),
    sharex=True,
    sharey=True,
    figsize=(18, 12),
)
for i, (clf, fair_comp) in enumerate(clf_comp.items()):
    for j, (k, v) in enumerate(fair_comp.items()):
        ax = axes[i, j]
        ax.plot(
            v.groupby("N")["Regret"].median(),
            label=k,
            c=METHOD_GROUPS[k]["color"],
            ls=METHOD_GROUPS[k]["linestyle"],
        )
        for _, per_dset in v.groupby("Dataset"):
            ax.plot(
                per_dset.groupby("N")["Regret"].mean(),
                alpha=0.05,
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
        ax.set_title(f"{clf} - {k}")
        ax.set_ylim(-1, 2)
        ax.grid()
# axes[0, -1].legend(ncol=2, fontsize=8, bbox_to_anchor=(1.02, 0.7))
fig.tight_layout()
fig.savefig("iclr-figures/rq1-dm-per-clf-indiv.pdf", bbox_inches="tight")


"""
Custom plot for comparing the data modes of distill methods
"""

boxprops = dict(
    boxstyle="round",
    facecolor="white",
    alpha=0.8,
    ec=None,
)
fig, axes = plt.subplots(
    nrows=4,
    ncols=4,
    sharex=True,
    sharey="row",
    figsize=(9, 5.5),
    layout="constrained",
)
for j, clf in enumerate(
    [
        "XGB",
        "KNN",
        "LR",
        "MLP",
    ]
):
    fair_comp = clf_comp[clf]
    for i, dm in enumerate(["KM", "AG", "GM", "KIP"]):
        ax = axes[j, i]
        ipc10 = {}
        for prefix in ["", " Enc.", " Rec."]:
            k = dm + prefix
            v = fair_comp[k]
            med_vals = v.groupby("N")["Regret"].median()
            ipc10[prefix] = med_vals.values[0]
            # if prefix:
            #     to_annot.append((prefix.strip(), med_vals.values[0]))
            ax.plot(
                med_vals,
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
        med_vals = v.groupby("N")["Regret"].median()
        # to_annot.append(("RND", med_vals.values[0]))
        ax.plot(
            med_vals,
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
        noenc = ipc10[""]
        enc = ipc10[" Enc."]
        rec = ipc10[" Rec."]
        enc_change = (noenc - enc) / noenc * 100
        rec_change = (noenc - rec) / noenc * 100
        print(
            f"{clf,dm}, enc: {enc_change:.2f}%, rec: {rec_change:.2f}%, rec-enc: {enc_change-rec_change :.2f}%"
        )
        ann_text = "Enc.:  "
        if enc_change > 0:
            ann_text += "+"
        ann_text += f"{enc_change:.2f}% | "
        ann_text += "Rec.: "
        if rec_change > 0:
            ann_text += "+"
        ann_text += f"{rec_change:.2f}%"
        ax.annotate(
            f"IPC=10\n{ann_text}",
            (55, 1.70),
            # xycoords="data",
            # xytext=(0.1, 0.1),
            # textcoords="offset points",
            # textcoords="axes fraction",
            bbox=boxprops,
            horizontalalignment="center",
            verticalalignment="top",
            fontsize=8,
            ha="center",
        )
        if j == 0:
            ax.set_title(f"{clf} - {dm}")
        else:
            ax.set_title(f"{clf} - {dm}")
        ax.set_xlim(7, 103)
        if j == 0:
            ax.set_ylim(0.25, 1.7)
        elif j == 1:
            ax.set_ylim(-0.25, 1.7)
        elif j == 2:
            ax.set_ylim(0.07, 1.7)
        else:
            ax.set_ylim(0, 1.7)
        ax.grid()
        if j == (3):
            axes[j, i].legend(
                ncol=2,
                fontsize=8,
                loc="lower right",
                bbox_to_anchor=(1.05, -0.80),
            )
        # ax.set_yticks(np.linspace(-0.5, 2, 11))
# fig.tight_layout()
fig.supxlabel("Instances Per Class")
fig.supylabel("Relative Regret")
fig.subplots_adjust(wspace=0.05, hspace=0.2)
fig.savefig("iclr-figures/rq1-dm-by-clf-small.pdf", bbox_inches="tight")


boxprops = dict(
    boxstyle="round",
    facecolor="white",
    alpha=0.8,
    ec=None,
)
fig, axes = plt.subplots(
    nrows=len(clf_comp),
    ncols=4,
    sharex=True,
    sharey="row",
    figsize=(12, 10),
    layout="constrained",
)
for j, clf in enumerate(
    [
        "XGB",
        "KNN",
        "LR",
        "MLP",
        "FTT",
        "RN",
    ]
):
    fair_comp = clf_comp[clf]
    for i, dm in enumerate(["KM", "AG", "GM", "KIP"]):
        ax = axes[j, i]
        to_annot = []
        for prefix in ["", " Enc.", " Rec."]:
            k = dm + prefix
            v = fair_comp[k]
            med_vals = v.groupby("N")["Regret"].median()
            to_annot.append((prefix.strip() if prefix else "N/A", med_vals.values[0]))
            ax.plot(
                med_vals,
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
        med_vals = v.groupby("N")["Regret"].median()
        # to_annot.append(("RND", med_vals.values[0]))
        ax.plot(
            med_vals,
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
        ann_text = "\n".join(f"{k}: {v:.4f}" for k, v in to_annot)
        ax.annotate(
            f"IPC=10\n{ann_text}",
            (103, 1.70),
            # xycoords="data",
            # xytext=(0.1, 0.1),
            # textcoords="offset points",
            # textcoords="axes fraction",
            bbox=boxprops,
            horizontalalignment="right",
            verticalalignment="top",
            fontsize=8,
        )
        if j == 0:
            ax.set_title(f"{clf} - {dm}")
        else:
            ax.set_title(f"{clf} - {dm}")
        ax.set_ylim(-0.25, 1.7)
        ax.set_xlim(7, 103)
        ax.grid()
        if j == len(clf_comp) - 1:
            axes[j, i].legend(
                ncol=2,
                fontsize=8,
                loc="lower right",
                bbox_to_anchor=(1, -0.55),
            )
        # ax.set_yticks(np.linspace(-0.5, 2, 11))
# fig.tight_layout()
fig.supxlabel("Instances Per Class")
fig.supylabel("Relative Regret")
fig.subplots_adjust(wspace=0.05, hspace=0.2)
fig.savefig("iclr-figures/rq1-dm-by-clf.pdf", bbox_inches="tight")


"""
Another custom plot. columns are classifiers, rows are data representations
"""

fig, axes = plt.subplots(
    nrows=3,
    ncols=len(clf_comp),
    sharex=True,
    sharey=True,
    figsize=(12, 6),
)
data_reprs = ["Original", "Encoded", "Reconstructed"]
postfixes = ["", " Enc.", " Rec."]
distill_methods = ["KIP", "GM", "AG", "KM"]
for i, (drname, dr) in enumerate(zip(data_reprs, postfixes)):
    for j, (clf, fair_comp) in enumerate(clf_comp.items()):
        ax = axes[i, j]
        for dm in distill_methods:
            k = dm + dr
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
        if i == 0:
            ax.set_title(clf)
        if j == 0:
            ax.set_ylabel(drname, size="large")
        ax.set_ylim(-1, 2)
        ax.grid()
        if i == 0 and j == len(clf_comp) - 1:
            axes[i, j].legend(ncol=2, fontsize=8)
fig.tight_layout()
fig.savefig("iclr-figures/rq1-dm-by-clf-repr.pdf", bbox_inches="tight")


# """
# RQ1: How beneficial are the learned representations for distillation?
# """
# print("RQ1: How beneficial are the learned representations for distillation?")
# for n in [10, 50, 100]:
#     table = []
#     for postfix in ["", " Enc.", " Rec."]:
#         table.append(
#             {
#                 dm: "{:.4f}".format(
#                     finalized[
#                         (finalized["N"] == n) & (finalized["Method"] == dm + postfix)
#                     ]["Regret"].mean()
#                 )
#                 for dm in ["KM", "AG", "GM", "KIP"]
#             }
#         )
#     table = pd.DataFrame(table)
#     table["RND"] = "{:.4f}".format(
#         finalized[(finalized["N"] == n) & (finalized["Method"] == "RND")][
#             "Regret"
#         ].mean()
#     )
#     table["Data Rep."] = ["Mixed", "Encoded", "Reconstructed"]
#     table = table[["Data Rep.", "RND", "KM", "AG", "GM", "KIP"]]
#     print(table.to_markdown(floatfmt=".4f"))
#     print()
#     with open(f"iclr-figures/rq1-compare-reg-n{n}.tex", "w") as f:
#         f.write(table.to_latex(index=False))


# get the final verdict to include in the abstract and discussion.


improvements = []
for (clf, n), grp in finalized2.groupby(["Classifier", "N"]):
    noenc = grp[grp["Method"] == "KM"]["Regret"].median()
    withenc = grp[grp["Method"] == "KM Enc."]["Regret"].median()
    improvement = (noenc - withenc) / noenc * 100
    if improvement < 0:
        continue
    improvements.append(improvement)
    print(
        f"{(clf, n)} from {noenc:.4f} to {withenc:.4f}, {improvement:.2f}% improvement"
    )
print(
    f"Average improvement: {np.mean(improvements):.2f}% ({np.min(improvements):.2f} - {np.max(improvements):.2f}%)"
)

# %%

"""
How much do the methods benefit?
"""

improvements = {method: [] for method in ["KM", "AG", "GM", "KIP"]}
for clf in clf_comp.keys():
    for method in ["KM", "AG", "GM", "KIP"]:
        ori = clf_comp[clf][method]
        enc = clf_comp[clf][method + " Enc."]
        ori_mean = ori[ori["N"] == 10]["Regret"].median()
        enc_mean = enc[enc["N"] == 10]["Regret"].median()
        if enc_mean < ori_mean:
            improvement = ((ori_mean - enc_mean) / ori_mean)
            improvements[method].append(improvement)
for method in ["KM", "AG", "GM", "KIP"]:
    _min, _max = min(improvements[method]), max(improvements[method])
    print(f"{method}: {_min*100:.2f} ~ {_max*100:.2f}")
