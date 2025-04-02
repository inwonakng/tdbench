import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import json
import scikit_posthocs as sp
import numpy as np

from tabdd.utils import progress_bar
from tabdd.config.paths import FIGURES_DIR
from tabdd.results.load import compute_groups, compute_ranks
from tabdd.results.plot.matplotlib import make_boxplot


def parse_tuple_or_str(tup_or_str):
    if isinstance(tup_or_str, tuple):
        return tup_or_str
    return json.loads(
        str(tup_or_str).replace("(", "[").replace(")", "]").replace("'", '"')
    )


def make_table(rank_per_mode: pd.DataFrame):
    avg_ranks = (
        rank_per_mode[["Target", "Rank"]]
        .groupby("Target")
        .mean()
        .sort_values("Rank")["Rank"]
    )

    table = pd.DataFrame(
        [
            {"Target": " ".join(parse_tuple_or_str(k)), "Rank": v}
            for k, v in avg_ranks.items()
        ]
    ).to_markdown(index=False)
    return table


def draw_compare_plot(rank_per_mode: pd.DataFrame, ax: plt.Axes):

    ranks = []
    for t, group in rank_per_mode.groupby("Target"):
        ranks.append(
            (
                "->".join(t).replace("mixed", "Mixed").replace("onehot", "Onehot"),
                group["Rank"].tolist(),
            )
        )
    ranks = sorted(ranks, key=lambda x: sum(x[1]) / len(x[1]))
    x, y = list(zip(*ranks))

    # fig, ax = plt.subplots(figsize=(6, 3))
    make_boxplot(
        x=x,
        y=y,
        ax=ax,
        rotate_xlabels=30,
        style="color",
    )


# helper funciton to parse large groups. Sometimes we only want to know the best performance
# in the group instead of the mean.
def get_group_aggr(aspects, direction: str = "max"):
    _aggr = max if direction == "max" else min

    def aggr_func(group, metric):
        return _aggr(gr[metric].mean() for _, gr in group.groupby(aspects))

    return aggr_func


figs_dir = FIGURES_DIR / "data_parse_mode"
figs_dir.mkdir(exist_ok=True, parents=True)

ranks_cache = Path("ranks")
ranks_cache.mkdir(exist_ok=True, parents=True)

ori_df = pd.read_csv("data_mode_switch_results_w_reg.csv")
ori_df["Encoder"] = ori_df["Encoder"].fillna("N/A")
ori_df["Distill Space"] = ori_df["Distill Space"].fillna("N/A")
ori_df["Distill Method"] = ori_df["Distill Method"].replace({"KMeans":"$k$-means", "Agglo": "Agglomerative"})

df = ori_df
# df = ori_df[~ori_df["Classifier"].isin(["FTTransformer", "ResNet"])]

# enc_stats = pd.read_csv("enc_stats.csv")
# ds_stats = pd.read_csv("ds_stats.csv")

print("# Q1: How does the data parse mode affect the performance?")

data_parse_rankings = []

# Onehot vs Mixed in all settings
groups = compute_groups(
    report=df[(df["Encoder"] == "N/A") & (df["Distill Method"] != "Random Sample")],
    targets=[
        "Data Parse Mode",
    ],
    metric="Score",
    exclude=[
        "Distill Space",
        "Encoder",
        "Convert Binary",
        "Output Space",
        "Post Data Parse Mode",
    ],
)
rankings = compute_ranks(groups, direction="max")
data_parse_rankings.append({"Algorithm": "Overall", **rankings.mean().to_dict()})

for distill_method in ["$k$-means", "Agglomerative", "KIP", "GM"]:
    groups = compute_groups(
        report=df[(df["Encoder"] == "N/A") & (df["Distill Method"] == distill_method)],
        targets=["Data Parse Mode"],
        metric="Score",
        exclude=[
            "Distill Space",
            "Encoder",
            "Convert Binary",
            "Output Space",
            "Post Data Parse Mode",
        ],
    )
    rankings = compute_ranks(groups, direction="max")
    data_parse_rankings.append(
        {"Algorithm": distill_method, **rankings.mean().to_dict()}
    )

data_parse_rankings = pd.DataFrame(data_parse_rankings).rename(
    columns={
        ("onehot",): "Binary",
        ("mixed",): "Original",
    }
)

special_dp_case = []

groups = compute_groups(
    report=df[
        (df["Data Parse Mode"] == "mixed")
        | ((df["Data Parse Mode"] == "onehot") & (df["Encoder"] != "NA"))
        & (df["Distill Method"] != "Random Sample")
    ],
    targets=[
        "Data Parse Mode",
    ],
    metric="Score",
    exclude=[
        "Distill Space",
        "Encoder",
        "Convert Binary",
        "Output Space",
        "Post Data Parse Mode",
    ],
    group_aggr="max",
)
rankings = compute_ranks(groups, direction="max")
special_dp_case.append({"Algorithm": "Overall", **rankings.mean().to_dict()})

for distill_method in ["$k$-means", "Agglomerative", "KIP", "GM"]:
    groups = compute_groups(
        report=df[
            (
                (df["Data Parse Mode"] == "mixed")
                | ((df["Data Parse Mode"] == "onehot") & (df["Encoder"] != "NA"))
            )
            & (df["Distill Method"] == distill_method)
        ],
        targets=["Data Parse Mode"],
        metric="Score",
        exclude=[
            "Distill Space",
            "Encoder",
            "Convert Binary",
            "Output Space",
            "Post Data Parse Mode",
        ],
        group_aggr="max",
    )
    rankings = compute_ranks(groups, direction="max")
    special_dp_case.append({"Algorithm": distill_method, **rankings.mean().to_dict()})

special_dp_case = pd.DataFrame(special_dp_case).rename(
    columns={
        ("onehot",): "Binary+Encoder",
        ("mixed",): "Original-Encoder",
    }
)

dm_rankings = data_parse_rankings.merge(special_dp_case, on="Algorithm")

print(
    dm_rankings.to_latex(
        float_format="%.4f",
        index=False,
    )
)

# quit()

print()
print()
print("# Q2: Which encoder is better? Do we even need an encoder?")

groups = compute_groups(
    report=df[(df["Data Parse Mode"] == "onehot") & (df["Distill Space"] != "N/A")],
    targets=[
        "Encoder",
    ],
    metric="Score",
    exclude=[
        "Output Space",
        "Convert Binary",
        "Distill Space",
        "Cluster Center",
    ],
    group_aggr=get_group_aggr(
        [
            "Output Space",
            "Convert Binary",
            "Distill Space",
            "Cluster Center",
        ]
    ),
)
rankings = compute_ranks(groups, direction="max")

print(f"## Encoder comparison")
print(rankings.mean().sort_values())


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
for encoder in ["TF", "GNN", "MLP"]:
    by_enc = df[
        df["Encoder"].str.contains(encoder)
        & (~df["Dataset"].isin(["CardioDisease", "InternetUsage"]))
        # & (~df["Classifier"].isin(["FTTransformer", "ResNet"]))
        & (df["Subset"] == "Test")
    ]
    param_sizes = []
    regrets = []
    for dataset, by_enc_ds in by_enc.groupby("Dataset"):
        param_size = enc_stats[
            (enc_stats["Model"].str.contains(encoder))
            & (enc_stats["Dataset"] == dataset)
        ]["Encoder Params"].values[0]
        plot_data += [
            {
                "Dataset": dataset,
                "Encoder": encoder,
                "Regret": reg,
                "Param Size": param_size,
            }
            for reg in by_enc_ds["Regret"].tolist()
        ]
plot_data = pd.DataFrame(plot_data)

fig, ax = plt.subplots(figsize=(4, 3))
for color, (encoder, group) in zip(colors, plot_data.groupby("Encoder")):
    x_med = np.median(group["Regret"])
    y_med = np.median(group["Param Size"])
    ax.scatter(x_med, y_med, color=color)
    x_left, x_right = np.quantile(group["Regret"], [0.25, 0.75])
    y_left, y_right = np.quantile(group["Param Size"], [0.25, 0.75])
    print(x_med, y_med)
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
ax.set_xlabel("Regret")
ax.set_ylabel("# Params")

# ax.set_yscale("log")
fig.savefig(figs_dir / "enc_param_vs_reg.pdf", bbox_inches="tight")

print()
print()
print("# Q3: Distill methods comparison")

oh = df[
    (df["Data Parse Mode"] == "onehot")
    & (df["Post Data Parse Mode"] == "onehot")
    & (df["Distill Method"].isin(["Random Sample", "$k$-means", "Agglomerative", "KIP", "GM"]))
    & (~df["Dataset"].isin(["CardioDisease", "InternetUsage"]))
    & (df["Subset"] == "Test")
    & (~df["Score"].isna())
    & (~df["Regret"].isna())
    & (~df["Scaled Regret"].isna())
    & (~df["Scaled Regret with RS"].isna())
]

to_group_by = [
    "N",
    "Dataset",
    "Encoder",
    "Distill Space",
    "Convert Binary",
    "Cluster Center",
    "Output Space",
    "Classifier",
]

dd_compare = []

with progress_bar() as progress:
    group_id = 0
    rs_groups = list(oh.groupby(["N", "Dataset", "Classifier"]))
    main_task = progress.add_task("By settings", total=len(rs_groups))
    for (n, ds, clf), rs_group in rs_groups:
        for (enc, dsp, cb, os), nocluster_grouped in rs_group.groupby(
            [
                "Encoder",
                "Distill Space",
                "Convert Binary",
                # "Cluster Center",
                "Output Space",
            ]
        ):
            for cc, grouped in nocluster_grouped.groupby("Cluster Center"):
                one_combo = []
                comps = dict(zip(to_group_by, [n, ds, enc, dsp, cb, cc, os, clf]))
                grouped = grouped.drop_duplicates()
                for dd, by_dd in grouped.groupby("Distill Method"):
                    if dd in ["Random Sample", "KIP", "GM"]:
                        continue
                    if len(by_dd) == 0:
                        continue
                    one_combo.append(
                        {
                            "Distill Method": dd,
                            "Group ID": group_id,
                            "Score": by_dd["Score"].mean(),
                            "Regret": by_dd["Regret"].mean(),
                            "Scaled Regret": by_dd["Scaled Regret"].mean(),
                            "Scaled Regret with RS": by_dd[
                                "Scaled Regret with RS"
                            ].mean(),
                            **comps,
                        }
                    )
                # special case for random sample
                rs_set = rs_group[
                    rs_group["Distill Method"] == "Random Sample"
                ].drop_duplicates()
                if len(rs_set) > 0:
                    one_combo.append(
                        {
                            "Distill Method": "Random Sample",
                            "Group ID": group_id,
                            "Score": rs_set["Score"].mean(),
                            "Regret": rs_set["Regret"].mean(),
                            "Scaled Regret": rs_set["Scaled Regret"].mean(),
                            "Scaled Regret with RS": rs_set[
                                "Scaled Regret with RS"
                            ].mean(),
                            **comps,
                        }
                    )
                # special case for gradient
                nocluster_set = nocluster_grouped[
                    nocluster_grouped["Distill Method"].isin(["KIP", "GM"])
                ].drop_duplicates()
                for dd, by_dd in nocluster_set.groupby("Distill Method"):
                    if len(nocluster_set) > 0:
                        one_combo.append(
                            {
                                "Distill Method": dd,
                                "Group ID": group_id,
                                "Score": by_dd["Score"].mean(),
                                "Regret": by_dd["Regret"].mean(),
                                "Scaled Regret": by_dd["Scaled Regret"].mean(),
                                "Scaled Regret with RS": by_dd[
                                    "Scaled Regret with RS"
                                ].mean(),
                                **comps,
                            }
                        )

                if len(one_combo) == 5:
                    dd_compare += one_combo
                    group_id += 1
        progress.update(main_task, advance=1)

dd_compare = pd.DataFrame(dd_compare)
# dd_compare["Distill Method"].replace({"KMeans": "$k$-means", "Agglo": "Agglomerative"}, inplace=True)
dd_compare.to_csv("distill_method_comparison.csv", index=False)

# rank here
dd_compare_ranks = compute_ranks(
    dd_compare,
    direction="min",
    target="Distill Method",
    metric="Scaled Regret",
)

print("## Distill Method Comparison")
print(dd_compare_ranks.mean().sort_values())

ww = sp.posthoc_wilcoxon(
    dd_compare,
    val_col="Scaled Regret",
    group_col="Distill Method",
)

fig, ax = plt.subplots(figsize=(4, 3))

sp.critical_difference_diagram(
    ranks=dd_compare_ranks.mean(),
    sig_matrix=ww,
    ax=ax,
)
fig.savefig(figs_dir / "distill_method_critical_diff.pdf", bbox_inches="tight")

to_group_by = [
    "N",
    "Dataset",
    "Distill Space",
    "Convert Binary",
    "Cluster Center",
    "Output Space",
    "Classifier",
]

dd_enc_compare = []

with progress_bar() as progress:
    group_id = 0
    rs_groups = list(oh.groupby(["N", "Dataset", "Classifier"]))
    main_task = progress.add_task("By settings", total=len(rs_groups))
    for (n, ds, clf), rs_grouped in oh.groupby(["N", "Dataset", "Classifier"]):
        for (dsp, cb, os), nocluster_grouped in rs_group.groupby(
            [
                "Distill Space",
                "Convert Binary",
                "Output Space",
            ]
        ):
            for cc, grouped in nocluster_grouped.groupby("Cluster Center"):
                one_combo = []
                comps = dict(zip(to_group_by, [n, ds, dsp, cb, cc, os, clf]))
                grouped = grouped.drop_duplicates()
                for (dd, enc), by_dd in grouped.groupby(["Distill Method", "Encoder"]):
                    if dd in ["Random Sample", "KIP", "GM"]:
                        continue
                    if len(by_dd) == 0:
                        continue
                    one_combo.append(
                        {
                            "Target": f"{dd}+{enc}",
                            "Distill Method": dd,
                            "Encoder": enc,
                            "Score": by_dd["Score"].mean(),
                            "Regret": by_dd["Regret"].mean(),
                            "Scaled Regret": by_dd["Scaled Regret"].mean(),
                            "Scaled Regret with RS": by_dd[
                                "Scaled Regret with RS"
                            ].mean(),
                            **comps,
                            "Group ID": group_id,
                        }
                    )
                # special case for random sample
                rs_set = rs_grouped[
                    rs_grouped["Distill Method"] == "Random Sample"
                ].drop_duplicates()
                if len(rs_set) > 0:
                    one_combo.append(
                        {
                            "Target": "Random Sample",
                            "Distill Method": "Random Sample",
                            "Group ID": group_id,
                            "Encoder": "N/A",
                            "Score": rs_set["Score"].mean(),
                            "Regret": rs_set["Regret"].mean(),
                            "Scaled Regret": rs_set["Scaled Regret"].mean(),
                            "Scaled Regret with RS": rs_set[
                                "Scaled Regret with RS"
                            ].mean(),
                            **comps,
                        }
                    )
                # special case for gradient
                nocluster_set = nocluster_grouped[
                    nocluster_grouped["Distill Method"].isin(["KIP", "GM"])
                ].drop_duplicates()
                for (dd, enc), by_dd in nocluster_set.groupby(
                    ["Distill Method", "Encoder"]
                ):
                    if len(nocluster_set) > 0:
                        one_combo.append(
                            {
                                "Target": f"{dd}+{enc}",
                                "Distill Method": dd,
                                "Group ID": group_id,
                                "Score": by_dd["Score"].mean(),
                                "Regret": by_dd["Regret"].mean(),
                                "Scaled Regret": by_dd["Scaled Regret"].mean(),
                                "Scaled Regret with RS": by_dd[
                                    "Scaled Regret with RS"
                                ].mean(),
                                **comps,
                            }
                        )

                if len(one_combo) == 4 * 6 + 1:
                    dd_enc_compare += one_combo
                    group_id += 1
        progress.update(main_task, advance=1)

dd_enc_compare = pd.DataFrame(dd_enc_compare)
dd_enc_compare.to_csv("distill_method_enc_comparison.csv", index=False)

# rank here

dd_enc_compare_ranks = compute_ranks(
    dd_enc_compare, direction="min", target="Target", metric="Scaled Regret"
)

print("## Distill Method and Encoder Comparison")
stats = []
for target, by_target in dd_enc_compare.groupby("Target"):
    if "MultiHead" not in target and target != "Random Sample":
        continue
    stats.append(
        {
            "Target": target,
            "Rank Min": dd_enc_compare_ranks[target].min(),
            "Rank Mean": dd_enc_compare_ranks[target].mean(),
            "Rank Median": dd_enc_compare_ranks[target].median(),
            "Rank Max": dd_enc_compare_ranks[target].max(),
            "Regret Min": by_target["Scaled Regret"].min(),
            "Regret Mean": by_target["Scaled Regret"].mean(),
            "Regret Median": by_target["Scaled Regret"].median(),
            "Regret Max": by_target["Scaled Regret"].max(),
        }
    )

print(pd.DataFrame(stats).to_latex(index=False, float_format="%.2f"))

print()
print()
print("# Q3.1: Performance across heterogeneous datasets")

balanced = (ds_stats["Minority Label Ratio"] > 0.4) & (
    ds_stats["Minority Label Ratio"] < 0.6
)
bal_datasets = ds_stats[balanced]["Dataset"]
imbal_datasets = ds_stats[~balanced]["Dataset"]

bal_ranks = compute_ranks(
    dd_compare[dd_compare["Dataset"].isin(bal_datasets)],
    direction="min",
    target="Distill Method",
    metric="Scaled Regret",
)

imbal_ranks = compute_ranks(
    dd_compare[dd_compare["Dataset"].isin(imbal_datasets)],
    direction="min",
    target="Distill Method",
    metric="Scaled Regret",
)

no_categ = ds_stats["Categorical Features"] == 0
all_categ = ds_stats["Categorical Features"] == ds_stats["Original Features"]
mix_categ = ~no_categ & ~all_categ
no_categ_datasets = ds_stats[no_categ]["Dataset"]
all_categ_datasets = ds_stats[all_categ]["Dataset"]
mix_categ_datasets = ds_stats[mix_categ]["Dataset"]

no_categ_ranks = compute_ranks(
    dd_compare[dd_compare["Dataset"].isin(no_categ_datasets)],
    direction="min",
    target="Distill Method",
    metric="Scaled Regret",
)

all_categ_ranks = compute_ranks(
    dd_compare[dd_compare["Dataset"].isin(all_categ_datasets)],
    direction="min",
    target="Distill Method",
    metric="Scaled Regret",
)

mix_categ_ranks = compute_ranks(
    dd_compare[dd_compare["Dataset"].isin(mix_categ_datasets)],
    direction="min",
    target="Distill Method",
    metric="Scaled Regret",
)


by_datasets = []
for dd in ["Random Sample", "$k$-means", "Agglomerative", "KIP", "GM"]:
    bal_ranks[dd]
    by_datasets.append({
        "Distill Method": dd,
        "Bal.": f"{bal_ranks[dd].mean():.2f} {{\\tiny({bal_ranks[dd].std():.2f})}} | {int(bal_ranks[dd].median())}",
        "Imbal.": f"{imbal_ranks[dd].mean():.2f} {{\\tiny({imbal_ranks[dd].std():.2f})}} | {int(imbal_ranks[dd].median())}",
        "No Cat.": f"{no_categ_ranks[dd].mean():.2f} {{\\tiny({no_categ_ranks[dd].std():.2f})}} | {int(no_categ_ranks[dd].median())}",
        "Mixed": f"{mix_categ_ranks[dd].mean():.2f} {{\\tiny({mix_categ_ranks[dd].std():.2f})}} | {int(mix_categ_ranks[dd].median())}",
        "All Cat.": f"{all_categ_ranks[dd].mean():.2f} {{\\tiny({all_categ_ranks[dd].std():.2f})}} | {int(all_categ_ranks[dd].median())}",
    })
by_datasets = pd.DataFrame(by_datasets)

by_datasets = pd.DataFrame(
    {
        "Bal.": bal_ranks.mean(),
        "Bal. Med.": bal_ranks.median(),
        "Imbal.": imbal_ranks.mean(),
        "Imbal. Med.": imbal_ranks.median(),
        "No Cat.": no_categ_ranks.mean(),
        "No Cat. Med.": no_categ_ranks.median(),
        "Mixed": mix_categ_ranks.mean(),
        "Mixed Med.": mix_categ_ranks.median(),
        "All Cat.": all_categ_ranks.mean(),
        "All Cat. Med.": all_categ_ranks.median(),
    }
)

print(by_datasets.to_latex(float_format="%.4f"))


print()
print()
print("# Q4: Which classifier benefits the most?")

groups = compute_groups(
    report=oh,
    targets=[
        "Classifier",
    ],
    metric="Regret",
)

groups["Target"] = groups["Target"].apply(lambda x: "+".join(x))


rankings = compute_ranks(groups, direction="min")
print(rankings.mean().sort_values())

ww = sp.posthoc_wilcoxon(
    groups,
    val_col="Regret Mean",
    group_col="Target",
)

fig, ax = plt.subplots(figsize=(4, 3))

sp.critical_difference_diagram(
    ranks=rankings.mean(),
    sig_matrix=ww,
    ax=ax,
)

fig.savefig(figs_dir / "classifier_critical_diff.pdf", bbox_inches="tight")


print()
print()
print("# Q5: How do different classifiers benefit?")

from tabdd.results.plot.matplotlib import make_boxplot

dd_perf_over_n = pd.concat(
    [
        compute_ranks(
            by_n_dd,
            direction="min",
            target="Distill Method",
            metric="Scaled Regret",
        ).assign(**{"N": n, "Classifier": clf, "Dataset": ds})
        for (n, ds, clf), by_n_dd in dd_compare.groupby(["N", "Dataset", "Classifier"])
    ]
)


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


ds_clf_n_rank = []
for (n, ds, clf, dd), by_ds_clf_n in dd_compare.groupby(
    ["N", "Dataset", "Classifier", "Distill Method"]
):
    rr = dd_compare_ranks.iloc[by_ds_clf_n["Group ID"]][dd].values
    ds_clf_n_rank += [
        {
            "N": n,
            "Dataset": ds,
            "Classifier": clf,
            "Distill Method": dd,
            "Rank": r,
            "Scaled Regret": sr,
            "Scaled Regret with RS": srr,
        }
        for r, (sr, srr) in zip(rr, by_ds_clf_n[["Scaled Regret", "Scaled Regret with RS"]].values)
    ]
ds_clf_n_rank = pd.DataFrame(ds_clf_n_rank)

import seaborn as sns
n_clfs = ds_clf_n_rank["Classifier"].nunique()

fig, axes = plt.subplots(figsize=(n_clfs*2, 1.8), ncols=n_clfs, sharey=True, sharex=True)
fig.subplots_adjust(hspace=0.22)
for i, (clf, by_clf) in enumerate(ds_clf_n_rank.groupby("Classifier")):
    sns.lineplot(
        data=by_clf,
        x="N",
        y="Rank",
        hue="Distill Method",
        # col = "Classifier",
        # kind="line",
        ax=axes.ravel()[i],
    )
    axes.ravel()[i].get_legend().remove()
    axes.ravel()[i].set_title(short_clf_name(clf))
# fig.delaxes(axes.ravel()[-1])
axes.ravel()[-1].legend(loc="center left", bbox_to_anchor=(1, 0.5))
fig.savefig(figs_dir / "clf_n_perf_rank.pdf", bbox_inches="tight")

fig, axes = plt.subplots(figsize=(n_clfs*2, 1.8), ncols=n_clfs, sharey=True, sharex=True)
fig.subplots_adjust(hspace=0.22)
for i, (clf, by_clf) in enumerate(ds_clf_n_rank.groupby("Classifier")):
    sns.lineplot(
        data=by_clf,
        x="N",
        y="Scaled Regret",
        hue="Distill Method",
        # col = "Classifier",
        # kind="line",
        ax=axes.ravel()[i],
    )
    axes.ravel()[i].get_legend().remove()
    axes.ravel()[i].set_title(short_clf_name(clf))
# fig.delaxes(axes.ravel()[-1])
axes.ravel()[-1].legend(loc="center left", bbox_to_anchor=(1, 0.5))
fig.savefig(figs_dir / "clf_n_perf_scaled_reg.pdf", bbox_inches="tight")

fig, axes = plt.subplots(figsize=(n_clfs*2, 1.8), ncols=n_clfs, sharey=True, sharex=True)
fig.subplots_adjust(hspace=0.22)
for i, (clf, by_clf) in enumerate(ds_clf_n_rank.groupby("Classifier")):
    sns.lineplot(
        data=by_clf,
        x="N",
        y="Scaled Regret",
        hue="Distill Method",
        estimator=np.median,
        # col = "Classifier",
        # kind="line",
        ax=axes.ravel()[i],
    )
    axes.ravel()[i].get_legend().remove()
    axes.ravel()[i].set_title(short_clf_name(clf))
# fig.delaxes(axes.ravel()[-1])
axes.ravel()[-1].legend(loc="center left", bbox_to_anchor=(1, 0.5))
fig.savefig(figs_dir / "clf_n_perf_scaled_reg_med.pdf", bbox_inches="tight")

print()
print()
print("# Q6. So which one is the best?")

from collections import Counter

top_n = 3
best_methods = []

for (n, ds, clf), overall_group in df[
    (df["Data Parse Mode"] == "onehot")
    & (df["Subset"] == "Test")
    & (~df["Distill Method"].isin(["Original"]))
].groupby(["N", "Dataset", "Classifier"]):
    best_methods.append(overall_group.sort_values(["Scaled Regret"])[:top_n])

best_methods = pd.concat(best_methods)
best_count = Counter(
    tuple(comb)
    for comb in best_methods[
        ["Encoder", "Distill Space", "Distill Method", "Output Space"]
    ].values
)

by_freq = sorted(best_count.items(), key=lambda x: x[1], reverse=True)

best_performers = []
for k, v in by_freq[:10]:
    best_performers.append(
        dict(
            zip(
                [
                    "Count",
                    "Encoder",
                    "Distill Space",
                    "Distill Method",
                    "Output Space",
                ],
                [v, *k],
            )
        )
    )

best_performers = pd.DataFrame(best_performers)
print(best_performers.to_latex(index=False))


# what's up with KIP?

has_kip = best_methods[
    (best_methods["Encoder"] == "TF-MultiHead")
    & (best_methods["Distill Space"] == "encoded")
    & (best_methods["Distill Method"] == "KIP")
    & (best_methods["Output Space"] == "encoded")
]

has_km = best_methods[
    (best_methods["Encoder"] == "TF-MultiHead")
    & (best_methods["Distill Space"] == "encoded")
    & (best_methods["Distill Method"] == "$k$-means")
    & (best_methods["Output Space"] == "encoded")
]

