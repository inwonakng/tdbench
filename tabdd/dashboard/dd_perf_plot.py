import pandas as pd
import plotly.graph_objects as go

from tabdd.results.load import compute_ranks
from tabdd.config.paths import DASHBOARD_ASSETS_DIR
from tabdd.dashboard.utils import (
    describe_df,
    save_table_json,
    dataset_names,
    distill_methods,
    dd_colors,
)

dd_compare = pd.read_csv("distill_method_comparison.csv")
dd_compare["Distill Method"] = dd_compare["Distill Method"].replace(
    {"$k$-means": "<i>k</i>-means"}
)

# rank here
dd_compare_ranks = compute_ranks(
    dd_compare,
    direction="min",
    target="Distill Method",
    metric="Scaled Regret",
)


fig = go.Figure()

# for dd in dd_compare_ranks.columns:
for dd,dd_color in zip(distill_methods, dd_colors):
    fig.add_trace(
        go.Box(
            y=dd_compare_ranks[dd],
            name=dd,
            marker=dict(color=dd_color),
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
    xaxis_title="Distill Method",
    yaxis_title="Downstream Performance Rank",
)

fig.write_html(DASHBOARD_ASSETS_DIR / f"plots/dd_perf_all_plot.html")

table = describe_df(dd_compare_ranks)
save_table_json(table, DASHBOARD_ASSETS_DIR / "tables/dd_perf_all_table.json")

for ds in dd_compare["Dataset"].unique():
    grouped = dd_compare[dd_compare["Dataset"] == ds]

    # rank here
    rankings = compute_ranks(
        grouped,
        direction="min",
        target="Distill Method",
        metric="Scaled Regret",
    )

    fig = go.Figure()

    # for dd in rankings.columns:
    for dd,dd_color in zip(distill_methods, dd_colors):
        fig.add_trace(
            go.Box(
                y=rankings[dd],
                name=dd,
                marker= dict(color=dd_color),
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
        xaxis_title="Distill Method",
        yaxis_title="Downstream Performance Rank",
    )

    fig.write_html(
        DASHBOARD_ASSETS_DIR / f"plots/dd_perf_{dataset_names[ds]}_plot.html"
    )

    table = describe_df(rankings)
    save_table_json(
        table, DASHBOARD_ASSETS_DIR / f"tables/dd_perf_{dataset_names[ds]}_table.json"
    )
