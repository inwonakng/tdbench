import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def plot_classifier_performance(
    distill_results: pd.DataFrame,
    title: str,
    colormap: dict,
):    
    data_subsets = distill_results['Subset'].unique().tolist()
    data_modes = distill_results['Data Mode'].unique().tolist()

    fig = make_subplots(
        rows = len(data_subsets),
        shared_xaxes=True,
        shared_yaxes=True,
        subplot_titles=data_subsets,
        horizontal_spacing=0.01,
        vertical_spacing=0.05,
    )

    for row, ds in enumerate(data_subsets):
        for dm in data_modes:
            subset = distill_results[
                (distill_results['Subset'] == ds) &
                (distill_results['Data Mode'] == dm)
            ]
            distill_size, score_mean, score_max, score_min = zip(*[
                (
                    s['N'].values[0],
                    s['Score'].mean(),
                    s['Score'].max(),
                    s['Score'].min(),
                ) for _,s in subset.groupby('N')
            ])
            fig.add_traces(
                [
                    go.Scatter(
                        x = distill_size,
                        y = score_mean,
                        line=dict(color = colormap[dm]['main']),
                        name = dm,
                        legendgroup = dm,
                        # legendgrouptitle_text = subset['Distill Group'].values[0], 
                        showlegend = row == 0,
                        mode='lines+markers',
                    ),
                    go.Scatter(
                        x = distill_size + distill_size[::-1],
                        y = score_max + score_min[::-1],
                        fill='toself',
                        fillcolor=colormap[dm]['fill'],
                        line=dict(color='rgba(255,255,255,0)'),
                        hoverinfo='skip',
                        showlegend=False,
                        name = dm,
                        # legendgroup = f'{dm}',
                        legendgroup = dm,
                    ),
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

