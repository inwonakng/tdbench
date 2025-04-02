import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .utils import add_alpha
from tabdd.models.utils import pretty_encoder_name

def plot_distilled_distribution(
    reduced_embs,
    reduce_method,
    colormap,
):
    distill_methods = reduced_embs['Output Mode'].unique().tolist()
    encoders = sorted(reduced_embs['Encoder'].unique().tolist(), key = pretty_encoder_name)[::-1]

    fig = make_subplots(
        rows = len(encoders),
        cols = len(distill_methods),
        # subplot_titles=[
        #     f'{dm} / {pretty_encoder_name(en)}'
        #     for en in encoders
        #     for dm in distill_methods
        # ],
        row_titles=[pretty_encoder_name(en) for en in encoders],
        column_titles = distill_methods,
        # col
        horizontal_spacing=0.03,
        vertical_spacing=0.03,
    )

    for row, encoder in enumerate(encoders):
        for col, distill_method in enumerate(distill_methods):
            sliced = reduced_embs[
                (reduced_embs['Encoder'] == encoder)
                & (reduced_embs['Output Mode'] == distill_method)
            ]
            for label, slice_by_label in sliced.groupby('Label'):        
                fig.add_trace(
                    go.Scatter(
                        x = slice_by_label['D1'],
                        y = slice_by_label['D2'],
                        mode = 'markers',
                        marker_color = add_alpha(colormap[int(label)],0.8),
                        hovertext = f'Label: {int(label)}',
                        hoverinfo = 'text',
                        legendgroup = label,
                        # showlegend = (row == 0 and col == 0),
                        showlegend=False,
                        name = f'Label: {int(label)}',
                    ),
                    row = row + 1,
                    col = col + 1
                )

    fig.update_layout(
        # title = f'Distilled Distribution [{reduce_method}]',
        # title_x = 0.5,
        height = 200 * len(encoders), 
        width = 200 * len(distill_methods),
        margin=dict(l=20, r=20, t=40, b=20),
        hoverlabel=dict(
            bgcolor="#ededed",
            font_size=12,
            namelength = -1,
        ),
    )

    return fig
