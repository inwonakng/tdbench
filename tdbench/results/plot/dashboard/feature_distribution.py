import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import math

def plot_feature_distribution(
    title: str,
    dataset: pd.DataFrame,
    labels: pd.Series,
    colormap: dict
):
    is_categorical = (dataset.dtypes == 'object').tolist()
    num_rows = math.ceil(dataset.shape[1]/4)
    nan_columns = dataset.columns[dataset.isna().any()]
    draw_nan = dataset.isna().any().any()
    unique_labels = labels.unique()
    # label_index = {
    #     l: i 
    #     for i,l in zip(
    #         labels.astype('category').cat.codes,
    #         unique_labels
    #     )
    # }
    label_index = {c:i for i, c in enumerate(labels.astype('category').cat.categories)}
    
    if draw_nan:
        fig = make_subplots(
            rows = num_rows + 1,
            cols = 4,
            specs= [[{},{},{},{}]] * num_rows + [
                [{'colspan':4}, None, None, None],
            ],
            subplot_titles = [
                f'{f} - {"cat" if is_cat else "cont"}'
                for f,is_cat in zip(dataset.columns.tolist(), is_categorical)
            ] + ['']*(4 -dataset.shape[1] % 4) + ['Invalid Values'],
            row_heights=[.25] * num_rows + [.1],
            horizontal_spacing=0.05,
            vertical_spacing=0.05,
        )

        fig.add_traces(
            [
                go.Histogram(
                    # x = dataset[feature],
                    y=dataset.isna().sum(0)[nan_columns],
                    # y=nan_columns,
                    orientation='h',
                    name = 'All',
                    legendgroup = 'All',
                    showlegend=True,
                    marker_color = colormap[-1],
                )
            ]
              + [
                go.Histogram(
                    # x = dataset[feature][labels == label],
                    y=dataset[labels == label].isna().sum(0)[nan_columns],
                    name = f'Label: {label}',
                    orientation='h',
                    legendgroup = label,
                    showlegend=True,
                    marker_color = colormap[label_index[label]],
                )
                for label in unique_labels
            ],
            rows = num_rows + 1,
            cols = 1,
        )
    else:
        fig = make_subplots(
            rows = num_rows,
            cols = 4,
            subplot_titles = [
                f'{f} - {"cat" if is_cat else "cont"}'
                for f,is_cat in zip(dataset.columns.tolist(), is_categorical)
            ],
            horizontal_spacing=0.05,
            vertical_spacing=0.05,
        )

    hide_xaxis = []
    for i, (feature, is_cat) in enumerate(zip(dataset.columns, is_categorical)):
        row = math.ceil((i+1)/4)
        col = 1 + i % 4
        fig.add_traces(
            [
                go.Histogram(
                    x = dataset[feature],
                    name = 'All',
                    legendgroup = 'All',
                    showlegend=False if draw_nan else (i == 0),
                    marker_color = colormap[-1],
                )
            ] + [
                go.Histogram(
                    x = dataset[feature][labels == label],
                    name = f'Label: {label}',
                    legendgroup = label,
                    showlegend=False if draw_nan else (i == 0),
                    marker_color = colormap[label_index[label]],
                )
                for label in unique_labels
            ],
            rows = row,
            cols = col,
        )
        if is_cat:
            hide_xaxis += [(row, col)]
    
    fig.update_layout(
        title = title,
        title_x = 0.5,
        height = 250 * num_rows + 100 * len(nan_columns), 
        width = 1200,
        margin=dict(l=20, r=20, t=80, b=20),
        hoverlabel=dict(
            bgcolor='#ededed',
            font_size=12,
            namelength = -1,
        ),
        yaxis_showticklabels = False,
        # barmode = 'overlay',
    )

    for row, col in hide_xaxis:
        fig.update_xaxes(showticklabels=False, row=row, col=col)

    return fig
