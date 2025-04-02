import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import math

def plot_tuned_parameters(
    tuned_parameters: pd.DataFrame,
    title: str,
    colormap: dict,
):    
    parameters = tuned_parameters['Param'].unique().tolist()
    data_modes = tuned_parameters['Data Mode'].unique().tolist()
    is_categorical = []

    for p in parameters:
        try:
            tuned_parameters[tuned_parameters['Param'] == p]['Value'].astype(float)
            is_categorical += [True]
        except:
            is_categorical += [False]

    num_rows = math.ceil(len(parameters)/4)
    fig = make_subplots(
        rows = num_rows,
        cols = 4,
        shared_xaxes=True,
        subplot_titles = parameters,
        horizontal_spacing=0.05,
        vertical_spacing=0.05,
    )

    # for row, dm in enumerate(data_modes):
    for i, (pa, is_cat) in enumerate(zip(parameters, is_categorical)):
        for dm in data_modes:
            row = math.ceil((i+1)/4)
            col = 1 + i % 4
            if is_cat:
                subset = tuned_parameters[
                    (tuned_parameters['Data Mode'] == dm) &
                    (tuned_parameters['Param'] == pa) 
                ]
                distill_size, value_mean, value_max, value_min = zip(*[
                    (
                        s['N'].values[0],
                        s['Value'].mean(),
                        s['Value'].max(),
                        s['Value'].min(),
                    ) for _,s in subset.groupby('N')
                ])
                fig.add_traces(
                    [
                        go.Scatter(
                            x = distill_size,
                            y = value_mean,
                            line=dict(color = colormap[dm]['main']),
                            name = dm,
                            legendgroup = dm,
                            # legendgrouptitle_text = subset['Distill Group'].values[0],
                            showlegend = i == 0,
                            mode='lines+markers',
                        ),
                        go.Scatter(
                            x = distill_size + distill_size[::-1],
                            y = value_max + value_min[::-1],
                            fill='toself',
                            fillcolor=colormap[dm]['fill'],
                            name = dm,
                            line=dict(color='rgba(255,255,255,0)'),
                            hoverinfo='skip',
                            showlegend=False,
                            legendgroup = dm,
                        ),
                    ],
                    rows = row,
                    cols = col,
                )

    fig.update_layout(
        title = title,
        title_x = 0.5,
        height = 250 * num_rows, 
        width = 1200,
        margin=dict(l=20, r=20, t=80, b=20),
        hovermode = 'x unified',
        hoverlabel=dict(
            bgcolor="#ededed",
            font_size=12,
            namelength = -1,
        ),
        # legend=dict(groupclick='toggleitem')
    )
    fig.update_traces(visible="legendonly")
    for trace in fig.data:
        if trace.name == data_modes[0]:
            trace.visible = True
    return fig

