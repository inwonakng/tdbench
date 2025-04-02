import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from tabdd.config.paths import RESULTS_CACHE_DIR

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


default_results = pd.read_csv(RESULTS_CACHE_DIR / "default_results.csv")

df = default_results[default_results["Subset"] == "Test"]

original = df[
    (df["Distill Method"] == "Original") & (df["Data Mode"] == "Mixed Original")
]

kmeans = df[
    (df["Distill Method"] == "KMeans")
    & (df["Data Mode"] == "KMeans-MLP-MultiHead-Encoded -> Encoded / Centroid")
]

kip = df[
    (df["Distill Method"] == "KIP")
    & (df["Data Mode"] == "KIP-MLP-MultiHead-Encoded -> Encoded / Centroid")
]

random_sample = df[(df["Distill Method"] == "Random Sample")]


res = pd.concat(
    [
        original,
        random_sample,
        kmeans,
        kip,
    ]
)

fig, axes = plt.subplots(figsize=(9, 5), nrows=2, ncols=3, sharex=True, sharey=True)

res = res.rename(columns={"N": "$n/L$"})
res["Classifier"] = res["Classifier"].apply(short_clf_name)
res["Distill Method"] = res["Distill Method"].apply(short_dm_name)

sns.lineplot(
    data=res,
    x="$n/L$",
    y="Score",
    hue="Distill Method",
    ax=axes.flatten()[0],
    errorbar=("ci", 95),
).set(title="Overall")
axes.flatten()[0].get_legend().remove()

for i, (clf, by_clf) in enumerate(res.groupby("Classifier")):
    sns.lineplot(
        data=by_clf,
        x="$n/L$",
        y="Score",
        hue="Distill Method",
        ax=axes.flatten()[i + 1],
        errorbar=("ci", 95),
    ).set(title=clf)
    # if i < (len(res["Classifier"].unique())-1):
    # if i > 0:
    if i != 1:
        axes.flatten()[i + 1].get_legend().remove()
# axes.flatten()[2].legend(loc="upper right", bbox_to_anchor=(1, -.5))
axes.flatten()[2].legend(loc="upper right", bbox_to_anchor=(1.6, 1.02))
for ax in axes.flat:
    for c in ax.collections:
        c.set_alpha(0.1)
fig.tight_layout()

fig.savefig("figures/best_perf_over_n.pdf", bbox_inches="tight")
