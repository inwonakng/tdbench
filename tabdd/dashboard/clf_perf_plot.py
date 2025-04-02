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
        for r, (sr, srr) in zip(
            rr, by_ds_clf_n[["Scaled Regret", "Scaled Regret with RS"]].values
        )
    ]
ds_clf_n_rank = pd.DataFrame(ds_clf_n_rank)


for clf, by_clf in ds_clf_n_rank.groupby("Classifier"):
    fig = go.Figure()
    # for dd, by_clf_dd in by_clf.groupby("Distill Method"):
    for dd,dd_color in zip(distill_methods, dd_colors):
        by_clf_dd = by_clf[by_clf["Distill Method"] == dd]
        fig.add_trace(
            go.Box(
                y=by_clf_dd["Rank"],
                name=dd,
                marker=dict(color=dd_color),
            ),
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

    fig.write_html(DASHBOARD_ASSETS_DIR / f"plots/clf_perf_all_{clf}_plot.html")

    rankings = pd.DataFrame(
        {dd: by_dd["Rank"].values for dd, by_dd in by_clf.groupby("Distill Method")}
    )
    table = describe_df(rankings)
    save_table_json(
        table, DASHBOARD_ASSETS_DIR / f"tables/clf_perf_all_{clf}_table.json"
    )

for ds, grouped in ds_clf_n_rank.groupby("Dataset"):
    for i, (clf, by_clf) in enumerate(grouped.groupby("Classifier")):
        fig = go.Figure()
        # for dd, by_clf_dd in by_clf.groupby("Distill Method"):

        for dd,dd_color in zip(distill_methods, dd_colors):
            by_clf_dd = by_clf[by_clf["Distill Method"] == dd]
            fig.add_trace(
                go.Box(
                    y=by_clf_dd["Rank"],
                    name=dd,
                    marker=dict(color=dd_color),
                ),
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
            DASHBOARD_ASSETS_DIR / f"plots/clf_perf_{dataset_names[ds]}_{clf}_plot.html"
        )

        rankings = pd.DataFrame(
            {dd: by_dd["Rank"].values for dd, by_dd in by_clf.groupby("Distill Method")}
        )
        table = describe_df(rankings)
        save_table_json(
            table,
            DASHBOARD_ASSETS_DIR
            / f"tables/clf_perf_{dataset_names[ds]}_{clf}_table.json",
        )
