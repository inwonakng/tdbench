import plotly.graph_objects as go
from plotly.subplots import make_subplots

from tabdd.models.utils import pretty_encoder_name
from .utils import add_alpha

def plot_encoded_distribution(
    reduced_embs,
    reduce_method,
    encoder_name,
    colormap,
):
    subsets = reduced_embs['Subset'].unique().tolist()
    spaces = reduced_embs['Space'].unique().tolist()

    fig = make_subplots(
        rows = len(subsets),
        cols = len(spaces),
        row_titles = subsets,
        column_titles = spaces,
        horizontal_spacing=0.05,
        vertical_spacing=0.05,
    )

    for row, subset in enumerate(subsets):
        for col, space in enumerate(spaces):
            sliced = reduced_embs[
                (reduced_embs['Subset'] == subset)
                & (reduced_embs['Space'] == space)
            ]
            for label, slice_by_label in sliced.groupby('Label'):        
                fig.add_trace(
                    go.Scatter(
                        x = slice_by_label['D1'],
                        y = slice_by_label['D2'],
                        mode = 'markers',
                        marker_color = add_alpha(colormap[int(label)], 0.8),
                        hovertext = f'Label: {int(label)}',
                        hoverinfo = 'text',
                        legendgroup = label,
                        showlegend = (row == 0 and col == 0),
                        name = f'Label: {int(label)}',
                    ),
                    row = row + 1,
                    col = col + 1
                )
    fig.update_layout(
        title = f'Dataset Distribution with {pretty_encoder_name(encoder_name)} [{reduce_method}]',
        title_x = 0.5,
        height = 600, 
        width = 1200,
        margin=dict(l=20, r=20, t=80, b=20),
        hoverlabel=dict(
            bgcolor="#ededed",
            font_size=12,
            namelength = -1,
        ),
    )

    return fig

