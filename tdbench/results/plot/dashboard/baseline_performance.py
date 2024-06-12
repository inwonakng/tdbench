import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def plot_baseline_performance(
    distill_results: pd.DataFrame,
    title: str,
    colormap: dict,
):
    data_subsets = distill_results['Subset'].unique().tolist()

    static_results = distill_results[
        ~distill_results['Data Mode'].str.contains('KMeans|Sample|KIP|Agglo')
        & (distill_results['N'] == 10)
    ]

    fig = make_subplots(
        rows = len(data_subsets),
        shared_xaxes=True,
        shared_yaxes=True,
        subplot_titles=data_subsets,
        horizontal_spacing=0.01,
        vertical_spacing=0.05,
    )

    for row, ds in enumerate(data_subsets):
        fig.add_traces(
            [
                go.Bar(
                    y = (dm,),
                    x = (score,),
                    marker_color = colormap[dm]['main'],
                    showlegend = row == 0,
                    legendgroup = dm,
                    name = dm,
                    orientation='h',
                    hoverinfo = 'x',
                )
                for dm, score in static_results[
                    (static_results['Subset'] == ds)
                ][[
                    'Data Mode', 'Score'
                ]].values[::-1]
            ],
            rows = row+1,
            cols = 1,
        )
    fig.update_layout(
        title = title,
        title_x = 0.5,
        height = 800, 
        width = 1000,
        margin=dict(l=40, r=20, t=80, b=20),
        hoverlabel=dict(
            bgcolor="#ededed",
            font_size=12,
            namelength = -1,
        ),
        legend_traceorder = 'reversed',
        # **{
        #     f'yaxis{i*2+1}': dict(
        #         range=[-.5,9.5],
        #     )
        #     for i in range(len(data_subsets))
        # },

        # **{
        #     f'xaxis{i+1}': dict(
        #         range=[0,1],
        #     )
        #     for i in range(len(train_modes))
        # },
    )

    return fig

