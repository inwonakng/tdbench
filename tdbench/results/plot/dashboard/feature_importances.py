import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def plot_feature_importances(
    original_feat_imps: pd.DataFrame, 
    distilled_feat_imps: pd.DataFrame, 
    title:str,
    colormap: dict,
    is_encoded: bool = False,
):
    original_data_mode = original_feat_imps['Data Mode'].values[0]
    distilled_data_mode = distilled_feat_imps['Data Mode'].values[0]
    fig = make_subplots(
        rows = 2,
        shared_xaxes=True,
        horizontal_spacing=0.01,
        vertical_spacing=0.08,
        subplot_titles=[
            original_data_mode, 
            distilled_data_mode,
        ]
    )

    if not is_encoded:
        fig.add_traces(
            [
                go.Bar(
                    y = [bf['Importance Score'].mean() for _, bf in values.groupby('Binary Feature')],
                    x = values['Binary Index'],
                    name = f'Feature {ori_f}',
                    legendgroup=ori_f,
                    marker_color = colormap[ori_f],
                )
                for ori_f, values in original_feat_imps.groupby('Original Feature')
            ],
            rows = 1,
            cols = 1,
        )
        fig.update_yaxes(type='log', row=1, col=1)

        fig.add_traces(
            [
                go.Bar(
                    y = [bf['Importance Score'].mean() for _, bf in values.groupby('Binary Feature')],
                    x = values['Binary Index'],
                    error_y = dict(
                        type='data',
                        array= [bf['Importance Score'].std(0) for _, bf in values.groupby('Binary Feature')],
                        color = '#696969',
                        thickness=0.7,
                    ),
                    name = f'Feature {ori_f}',
                    legendgroup=ori_f,
                    marker_color = colormap[ori_f],
                    showlegend= False,
                )
                for ori_f, values in distilled_feat_imps.groupby('Original Feature')
            ],
            rows = 2,
            cols = 1,
        )
        fig.update_yaxes(type='log', row=3, col=1)

    else:
        fig.add_traces(
            [
                go.Bar(
                    y = [values['Importance Score'].mean()],
                    x = [f],
                    error_y = dict(
                        type='data',
                        array= [values['Importance Score'].std()],
                        color = '#696969',
                        thickness=0.7,
                    ),
                    name = f'Feature {f}',
                    legendgroup=f,
                    marker_color = colormap[f],
                )
                for f,values in original_feat_imps.groupby('Binary Index')
            ],
            rows = 1,
            cols = 1,
        )
        fig.update_yaxes(type='log', row=1, col=1)

        fig.add_traces(
            [
                go.Bar(
                    y = [values['Importance Score'].mean()],
                    x = [f],
                    error_y = dict(
                        type='data',
                        array= [values['Importance Score'].std()],
                        color = '#696969',
                        thickness=0.7,
                    ),
                    name = f'Feature {f}',
                    legendgroup=f,
                    marker_color = colormap[f],
                    showlegend=False,
                )
                for f,values in distilled_feat_imps.groupby('Binary Index')
            ],
            rows = 2,
            cols = 1,
        )
        fig.update_yaxes(type='log', row=3, col=1)

    fig.update_layout(
        title = title,
        title_x = 0.5,
        height = 400, 
        width = 1000,
        margin=dict(l=40, r=20, t=80, b=20),
        hoverlabel=dict(
            bgcolor="#ededed",
            font_size=12,
            namelength = -1,
        ),
        bargap=0.1,
        bargroupgap=0.2,
    )
    return fig
