import pandas as pd

from tabdd.results.load import compute_groups, compute_ranks
from tabdd.config.paths import DASHBOARD_ASSETS_DIR

from tabdd.dashboard.utils import (
    get_group_aggr,
    describe_df,
    save_table_json,
    dataset_names,
    encoders,
    enc_colors,
)

ori_df = pd.read_csv("data_mode_switch_results_w_reg.csv")
ori_df["Encoder"] = ori_df["Encoder"].fillna("N/A")
ori_df["Distill Space"] = ori_df["Distill Space"].fillna("N/A")
ori_df["Distill Method"] = ori_df["Distill Method"].replace(
    {"KMeans": "$k$-means", "Agglo": "Agglomerative"}
)

df = ori_df

enc_stats = pd.read_csv("enc_stats.csv")
ds_stats = pd.read_csv("ds_stats.csv")

pretty_enc_names = {
    ("GNN",): "GNN",
    ("GNN-MultiHead",): "GNN-SFT",
    ("MLP",): "MLP",
    ("MLP-MultiHead",): "MLP-SFT",
    ("TF",): "TF",
    ("TF-MultiHead",): "TF-SFT",
    ("N/A",): "N/A",
}

# all datasets first

enc_compare = compute_groups(
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

rankings = compute_ranks(enc_compare, direction="max").rename(columns=pretty_enc_names)
# rankings.mean().sort_values()

import plotly.graph_objects as go

fig = go.Figure()

for enc, enc_color in zip(encoders, enc_colors):
    fig.add_trace(
        go.Box(
            y=rankings[enc],
            name=enc,
            marker=dict(color=enc_color),
        )
    )

fig.update_layout(
    barmode="stack",
    margin=dict(
        l=0,
        r=0,
        t=0,
        b=0,
    ),
    hoverlabel=dict(
        bgcolor="#ededed",
        font_size=12,
        namelength=-1,
    ),
    xaxis_title="Encoder Architecture",
    yaxis_title="Downstream Performance Rank",
)
fig.write_html(DASHBOARD_ASSETS_DIR / "plots/enc_perf_all_plot.html")


table = describe_df(rankings)
save_table_json(table, DASHBOARD_ASSETS_DIR / "tables/enc_perf_all_table.json")

for ds in df["Dataset"].unique():
    groups = compute_groups(
        report=df[
            (df["Data Parse Mode"] == "onehot")
            & (df["Distill Space"] != "N/A")
            & (df["Dataset"] == ds)
        ],
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

    rankings = compute_ranks(groups, direction="max").rename(columns=pretty_enc_names)

    fig = go.Figure()

    for enc, enc_color in zip(encoders, enc_colors):
        fig.add_trace(
            go.Box(
                y=rankings[enc],
                name=enc,
                marker=dict(color=enc_color),
            )
        )

    fig.update_layout(
        barmode="stack",
        margin=dict(
            l=0,
            r=0,
            t=0,
            b=0,
        ),
        hoverlabel=dict(
            bgcolor="#ededed",
            font_size=12,
            namelength=-1,
        ),
        xaxis_title="Encoder Architecture",
        yaxis_title="Downstream Performance Rank",
    )

    fig.write_html(
        DASHBOARD_ASSETS_DIR / f"plots/enc_perf_{dataset_names[ds]}_plot.html"
    )

    table = describe_df(rankings)
    save_table_json(
        table, DASHBOARD_ASSETS_DIR / f"tables/enc_perf_{dataset_names[ds]}_table.json"
    )
