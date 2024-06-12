import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

def plot_classifier_performance_overall(
    distill_results: pd.DataFrame,
    title: str,
    colormap: dict,
):    
    data_subsets = distill_results['Subset'].unique().tolist()
    data_modes = distill_results['Data Mode'].unique().tolist()
    distill_sizes = sorted(distill_results['N'].unique().tolist())
    datasets = sorted(distill_results['Dataset'].unique().tolist())

    fig = make_subplots(
        rows = len(data_subsets),
        shared_xaxes=True,
        shared_yaxes=True,
        subplot_titles=data_subsets,
        horizontal_spacing=0.01,
        vertical_spacing=0.05,
    )
    # print(title)
    for row, ds in enumerate(data_subsets):
        subset = distill_results[
            (distill_results['Subset'] == ds)
        ]
        for dm in data_modes:
            # distill_size = []
            score_mean = []
            score_max = []
            score_min = []
            for N in distill_sizes:
                means, maxs, mins = [],[],[]
                for dataset in datasets:
                    by_dataset = subset[(subset['N'] == N) & (subset['Dataset'] == dataset)]
                    # check if we have results. if not skip
                    if (
                        len(by_dataset[by_dataset['Data Mode'] == dm]) > 0
                        and len(by_dataset[by_dataset['Data Mode'] == 'Mixed Original']) > 0
                        and len(by_dataset[by_dataset['Data Mode'] == 'Random Sample']) > 0
                    ):
                        target = by_dataset[by_dataset['Data Mode'] == dm]['Score'].mean()
                        random = by_dataset[by_dataset['Data Mode'] == 'Random Sample']['Score'].mean()
                        baseline = by_dataset[by_dataset['Data Mode'] == 'Mixed Original']['Score'].values[0]
                        # print(target - random, baseline-random)
                        # if 0 == (baseline - random):
                            # print('we have a zero???',baseline, random,  dataset, dm)
                        # means += [(target - random) / (baseline - random)]
                        means += [target]
                score_mean += [np.mean(means)]
                score_max += [np.max(means)]
                score_min += [np.min(means)]

            fig.add_traces(
                [
                    go.Scatter(
                        x = distill_sizes,
                        y = score_mean,
                        line=dict(color = colormap[dm]['main']),
                        name = dm,
                        legendgroup = dm,
                        # legendgrouptitle_text = subset['Distill Group'].values[0], 
                        showlegend = (row == 0),
                        mode='lines+markers',
                    ),
                    # go.Scatter(
                    #     x = distill_sizes + distill_sizes[::-1],
                    #     y = score_max + score_min[::-1],
                    #     fill='toself',
                    #     fillcolor=colormap[dm]['fill'],
                    #     line=dict(color='rgba(255,255,255,0)'),
                    #     hoverinfo='skip',
                    #     showlegend=False,
                    #     name = dm,
                    #     legendgroup = dm,
                    # ),
                ],
                rows = row+1,
                cols = 1
            )

    fig.update_layout(
        title = title,
        title_x = 0.5,
        height = 800, 
        width = 1200,
        margin=dict(l=20, r=20, t=80, b=20),
        hovermode = 'x unified',
        hoverlabel=dict(
            bgcolor="#ededed",
            font_size=12,
            namelength = -1,
        ),
        # legend=dict(groupclick='toggleitem')
        # legend=dict(
        #     y = -10,
        #     yref = 'container',
        # )
    )

    fig.update_traces(visible="legendonly")
    for trace in fig.data:
        if trace.name in [data_modes[0], 'Random Sample']:
            trace.visible = True

    return fig

