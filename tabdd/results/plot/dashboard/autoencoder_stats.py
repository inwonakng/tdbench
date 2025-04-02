import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from tabdd.models.utils import pretty_encoder_name

def plot_autoencoder_stats(
    encoder_stats: pd.DataFrame,
    colormap: dict,
):
    encoders = encoder_stats['Model'].unique().tolist()
    fig = make_subplots(
        rows = 2,
        cols = 3,
        shared_yaxes=True,
        horizontal_spacing=0.01,
        vertical_spacing=0.1,
        specs = [
            [{},{},{}],
            [{'colspan':3},None,None],
        ],
        subplot_titles=[
            'Train Accuracy',
            'Val Accuracy',
            'Test Accuracy',
            'Network Size',
        ]
    )

    for i,en in enumerate(encoders):
        sliced = encoder_stats[encoder_stats['Model'] == en]
        fig.add_trace(
            go.Bar(
                x = sliced['Train Recon Accuracy'][::-1],
                y = sliced['Dataset'][::-1],
                marker_color = colormap[en]['score'],
                legendgroup = pretty_encoder_name(en),
                name = pretty_encoder_name(en),
                orientation='h',
                hoverinfo = 'x',
                legendrank = i,
            ),
            row = 1,
            col = 1,
        )

        fig.add_trace(
            go.Bar(
                x = sliced['Val Recon Accuracy'][::-1],
                y = sliced['Dataset'][::-1],
                marker_color = colormap[en]['score'],
                showlegend = False,
                legendgroup = pretty_encoder_name(en),
                name = pretty_encoder_name(en),
                orientation='h',
                hoverinfo = 'x',
            ),
            row = 1,
            col = 2,
        )

        fig.add_trace(
            go.Bar(
                x = sliced['Test Recon Accuracy'][::-1],
                y = sliced['Dataset'][::-1],
                marker_color = colormap[en]['score'],
                showlegend = False,
                legendgroup = pretty_encoder_name(en),
                name = pretty_encoder_name(en),
                orientation='h',
                hoverinfo = 'x',
            ),
            row = 1,
            col = 3,
        )

        fig.add_traces(
            [
                go.Bar(
                    y = sliced['Dataset'][::-1],
                    x = (sliced['Encoder Params'] + sliced['Decoder Params'])[::-1],
                    marker_color = colormap[en]['decoder'],
                    legendgroup = f'{en} Decoder',
                    name = f'{pretty_encoder_name(en)} Decoder Parameters',
                    orientation='h',
                    hoverinfo = 'text',
                    hovertext=[
                        f'{d} / {d+e}'
                        for d,e in zip(
                            sliced['Decoder Params'][::-1],
                            sliced['Encoder Params'][::-1],
                        )
                    ],
                    offsetgroup=en,
                    legendrank=999,
                ),
                go.Bar(
                    y = sliced['Dataset'][::-1],
                    x = sliced['Encoder Params'][::-1],
                    marker_color = colormap[en]['encoder'],
                    legendgroup = f'{en} Encoder',
                    name = f'{pretty_encoder_name(en)} Encoder Parameters',
                    orientation='h',
                    hoverinfo = 'text',
                    hovertext=[
                        f'{d} / {d+e}'
                        for d,e in zip(
                            sliced['Decoder Params'][::-1],
                            sliced['Encoder Params'][::-1],
                        )
                    ],
                    offsetgroup=en,
                    legendrank=998,
                ),
            ],
            rows = 2,
            cols = 1
        )

    fig.update_layout(
        title = 'AutoEncoder statistics',
        title_x = 0.5,
        height = 500, 
        width = 1000,
        margin=dict(l=40, r=20, t=80, b=20),
        hoverlabel=dict(
            bgcolor="#ededed",
            font_size=12,
            namelength = -1,
        ),
        barmode='group',
        bargap=0.1,
        bargroupgap=0.1,
    )
    return fig


