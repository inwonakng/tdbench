import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def plot_dataset_stats(dataset_stats,colormap):
    fig = make_subplots(
        rows = 2,
        subplot_titles=[
            'Label Ratio',
            'Feature Ratio',
        ],
        horizontal_spacing=0.05,
        vertical_spacing=0.07,
    )

    label_ratio = np.zeros((dataset_stats['Labels'].max(),len(dataset_stats)))
    for i,row in enumerate(dataset_stats['Label Ratio'].values):
        for j,val in enumerate(row):
            label_ratio[j,i] = val
    stacked_label_ratio = label_ratio.cumsum(0)

    fig.add_traces(
        [
            go.Bar(
                y = dataset_stats['Dataset'][::-1],
                x = stacked[::-1],
                marker_color = colormap['labels'][label],
                legendgroup = str(label),
                name = f'Label: {label}',
                orientation='h',
                hoverinfo = 'text',
                hovertext=[
                    f'{int(c)}'
                    for c in count[::-1]
                ],
                offsetgroup=0,
                legendrank= label
            )
            for label, count, stacked, total_count in zip(
                np.arange(label_ratio.shape[0])[::-1],
                label_ratio[::-1],
                stacked_label_ratio[::-1],
                dataset_stats['Rows'][::-1]
            )
        ],
        rows = 1,
        cols = 1
    )

    fig.add_traces(
        [
            go.Bar(
                y = dataset_stats['Dataset'][::-1],
                x = (
                    dataset_stats['Continuous Features']
                    + dataset_stats['Categorical Features']
                )[::-1],
                marker_color = colormap['features']['categorical'],
                legendgroup = 'Categorical',
                name = f'Categorical Features',
                orientation='h',
                hoverinfo = 'text',
                hovertext=[
                    f'{cat} / {cat+cont}'
                    for cat, cont in 
                    zip(
                        dataset_stats['Categorical Features'][::-1],
                        dataset_stats['Continuous Features'][::-1],
                    )
                ],
                offsetgroup=1,
                legendrank=999,
            ),
            go.Bar(
                y = dataset_stats['Dataset'][::-1],
                x = dataset_stats['Continuous Features'][::-1],
                marker_color = colormap['features']['continuous'],
                legendgroup = 'Continuous',
                name = f'Continuous Features',
                orientation='h',
                hoverinfo = 'text',
                hovertext=[
                    f'{cont} / {cat+cont}'
                    for cat, cont in 
                    zip(
                        dataset_stats['Categorical Features'][::-1],
                        dataset_stats['Continuous Features'][::-1],
                    )
                ],
                offsetgroup=1,
                legendrank=998,
            ),
        ],
        rows = 2,
        cols = 1
    )

    fig.update_layout(
        title = 'Dataset Statistics',
        title_x = 0.5,
        height = 600, 
        width = 1000,
        margin=dict(l=40, r=20, t=80, b=20),
        hoverlabel=dict(
            bgcolor="#ededed",
            font_size=12,
            namelength = -1,
        ),
    )

    return fig

