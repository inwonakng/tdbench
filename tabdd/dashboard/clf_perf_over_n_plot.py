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
    # for dd, by_dd in by_clf.groupby("Distill Method"):
    for dd, dd_color in zip(distill_methods[::-1], dd_colors[::-1]):
        by_dd = by_clf[by_clf["Distill Method"] == dd]
        x, y, ci_upper, ci_lower = [], [], [], []

        for n, by_n in by_dd.groupby("N"):
            ci = by_n["Rank"].sem() * 1.96
            x.append(n)
            y.append(by_n["Rank"].mean())
            ci_lower.append(by_n["Rank"].mean() - ci)
            ci_upper.append(by_n["Rank"].mean() + ci)

        fig.add_traces(
            [
                go.Scatter(
                    name=f"{dd} Average",
                    x=x,
                    y=y,
                    mode="lines",
                    line=dict(color=dd_color),
                ),
                go.Scatter(
                    name=f"{dd} 95% CI Upper",
                    x=x,
                    y=ci_upper,
                    mode="lines",
                    marker=dict(color=dd_color),
                    line=dict(width=0),
                    showlegend=False,
                ),
                go.Scatter(
                    name=f"{dd} 95% CI Lower",
                    x=x,
                    y=ci_lower,
                    marker=dict(color=dd_color),
                    line=dict(width=0),
                    mode="lines",
                    fillcolor=dd_color.replace("1.0", "0.1"),
                    fill="tonexty",
                    showlegend=False,
                ),
            ]
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
        hovermode="x",
        xaxis_title="Distilled Data Size (<i>n</i>)",
        yaxis_title="Distill Method Rank",
    )

    fig.update_yaxes(rangemode="tozero")
    fig.write_html(DASHBOARD_ASSETS_DIR / f"plots/clf_perf_over_n_all_{clf}_plot.html")

    for ds, by_ds in by_clf.groupby("Dataset"):
        fig = go.Figure()
        # for dd, by_dd in by_ds.groupby("Distill Method"):
        for dd, dd_color in zip(distill_methods[::-1], dd_colors[::-1]):
            by_dd = by_ds[by_ds["Distill Method"] == dd]
            x, y, ci_upper, ci_lower = [], [], [], []

            for n, by_n in by_dd.groupby("N"):
                ci = by_n["Rank"].sem() * 1.96
                x.append(n)
                y.append(by_n["Rank"].mean())
                ci_lower.append(by_n["Rank"].mean() - ci)
                ci_upper.append(by_n["Rank"].mean() + ci)

            fig.add_traces(
                [
                    go.Scatter(
                        name=f"{dd} Average",
                        x=x,
                        y=y,
                        mode="lines",
                        line=dict(color=dd_color),
                    ),
                    go.Scatter(
                        name=f"{dd} 95% CI Upper",
                        x=x,
                        y=ci_upper,
                        mode="lines",
                        marker=dict(color=dd_color),
                        line=dict(width=0),
                        showlegend=False,
                    ),
                    go.Scatter(
                        name=f"{dd} 95% CI Lower",
                        x=x,
                        y=ci_lower,
                        marker=dict(color=dd_color),
                        line=dict(width=0),
                        mode="lines",
                        fillcolor=dd_color.replace("1.0", "0.1"),
                        fill="tonexty",
                        showlegend=False,
                    ),
                ]
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
            hovermode="x",
            xaxis_title="Distilled Data Size (<i>n</i>)",
            yaxis_title="Distill Method Rank",
        )

        fig.update_yaxes(rangemode="tozero")
        fig.write_html(
            DASHBOARD_ASSETS_DIR
            / f"plots/clf_perf_over_n_{dataset_names[ds]}_{clf}_plot.html"
        )
