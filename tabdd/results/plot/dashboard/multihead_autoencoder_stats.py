import plotly.graph_objects as go
from plotly.subplots import make_subplots

from tabdd.models.utils import pretty_encoder_name

def plot_multihead_autoencoder_stats(
    encoder_stats,
    colormap,
):
    encoders = encoder_stats['Model'].unique().tolist()
    fig = make_subplots(
        rows = 3,
        cols = 3,
        specs=[
            [{},{},{}],
            [{},{},{}],
            [{'colspan':3}, None, None],
        ],
        shared_yaxes=True,
        horizontal_spacing=0.01,
        vertical_spacing=0.1,
        subplot_titles=[
            'Train Recon Accuracy',
            'Val Recon Accuracy',
            'Test Recon Accuracy',
            'Train Predict Accuracy',
            'Val Predict Accuracy',
            'Test Predict Accuracy',
            'Network Size',
        ]
    )

    for i,en in enumerate(encoders):
        sliced = encoder_stats[encoder_stats['Model'] == en]
        fig.add_trace(
            go.Bar(
                x = sliced['Train Recon Accuracy'][::-1],
                y = sliced['Dataset'][::-1],
                marker_color = colormap[en]['recon_score'],
                legendgroup = f'{en} recon',
                name = f'{pretty_encoder_name(en)} Recon Accuracy',
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
                marker_color = colormap[en]['recon_score'],
                showlegend = False,
                legendgroup = f'{en} recon',
                name = f'{pretty_encoder_name(en)} Recon Accuracy',
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
                marker_color = colormap[en]['recon_score'],
                showlegend = False,
                legendgroup = f'{en} recon',
                name = f'{pretty_encoder_name(en)} Recon Accuracy',
                orientation='h',
                hoverinfo = 'x',
            ),
            row = 1,
            col = 3,
        )

        fig.add_trace(
            go.Bar(
                x = sliced['Train Predict Accuracy'][::-1],
                y = sliced['Dataset'][::-1],
                marker_color = colormap[en]['predict_score'],
                legendgroup = f'{en} predict',
                name = f'{pretty_encoder_name(en)} Predict Accuracy',
                orientation='h',
                hoverinfo = 'x',
                legendrank = i,
            ),
            row = 2,
            col = 1,
        )

        fig.add_trace(
            go.Bar(
                x = sliced['Val Predict Accuracy'][::-1],
                y = sliced['Dataset'][::-1],
                marker_color = colormap[en]['predict_score'],
                showlegend = False,
                legendgroup = f'{en} predict',
                name = f'{pretty_encoder_name(en)} Predict Accuracy',
                orientation='h',
                hoverinfo = 'x',
            ),
            row = 2,
            col = 2,
        )

        fig.add_trace(
            go.Bar(
                x = sliced['Test Predict Accuracy'][::-1],
                y = sliced['Dataset'][::-1],
                marker_color = colormap[en]['predict_score'],
                showlegend = False,
                legendgroup = f'{en} predict',
                name = f'{pretty_encoder_name(en)} Predict Accuracy',
                orientation='h',
                hoverinfo = 'x',
            ),
            row = 2,
            col = 3,
        )

        fig.add_traces(
            [
                go.Bar(
                    y = sliced['Dataset'][::-1],
                    x = (sliced['Encoder Params'] + sliced['Decoder Params'] + sliced['Classifier Params'])[::-1],
                    marker_color = colormap[en]['classifier'],
                    legendgroup = f'{en} Classifier',
                    name = f'{pretty_encoder_name(en)} Classifier Parameters',
                    orientation='h',
                    hoverinfo = 'text',
                    hovertext=[
                        f'{d} / {d+e+c}'
                        for d,e,c in zip(
                            sliced['Decoder Params'][::-1],
                            sliced['Encoder Params'][::-1],
                            sliced['Classifier Params'][::-1],
                        )
                    ],
                    offsetgroup=en,
                    legendrank=999,
                ),
                go.Bar(
                    y = sliced['Dataset'][::-1],
                    x = (sliced['Encoder Params'] + sliced['Decoder Params'])[::-1],
                    marker_color = colormap[en]['decoder'],
                    legendgroup = f'{en} Decoder',
                    name = f'{pretty_encoder_name(en)} Decoder Parameters',
                    orientation='h',
                    hoverinfo = 'text',
                    hovertext=[
                        f'{d} / {d+e+c}'
                        for d,e,c in zip(
                            sliced['Decoder Params'][::-1],
                            sliced['Encoder Params'][::-1],
                            sliced['Classifier Params'][::-1],
                        )
                    ],
                    offsetgroup=en,
                    legendrank=998,
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
                        f'{d} / {d+e+c}'
                        for d,e,c in zip(
                            sliced['Decoder Params'][::-1],
                            sliced['Encoder Params'][::-1],
                            sliced['Classifier Params'][::-1],
                        )
                    ],
                    offsetgroup=en,
                    legendrank=997,
                ),
                
            ],
            rows = 3,
            cols = 1
        )

    fig.update_layout(
        title = 'MutiHeadAutoEncoder statistics',
        title_x = 0.5,
        height = 700, 
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

