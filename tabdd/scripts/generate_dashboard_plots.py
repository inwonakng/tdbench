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

df = pd.read_csv("data_mode_switch_results_w_reg.csv")
df["Encoder"] = df["Encoder"].fillna("N/A")
df["Distill Space"] = df["Distill Space"].fillna("N/A")
df["Distill Method"] = df["Distill Method"].replace(
    {"KMeans": "$k$-means", "Agglo": "Agglomerative"}
)
df = df[
    (df["Subset"] == "Test") & (~df["Classifier"].isin(["FTTransformer", "ResNet"]))
]

enc_stats = pd.read_csv("enc_stats.csv")
ds_stats = pd.read_csv("ds_stats.csv")


import plotly.graph_objects as go

fig = go.Figure(
    data = [
        go.Table(
            header=dict(values=["A Scores", "B Scores"]),
            cells = dict(values=[[100, 90, 80, 90], [95, 85, 75, 95]]),
        )
    ],
)

fig.write_json("test_table.json")
