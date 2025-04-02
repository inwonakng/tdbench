from omegaconf import OmegaConf
import hydra
import pandas as pd
import json
import yaml
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from tabdd.data import TabularDataModule
from tabdd.config.paths import DASHBOARD_ASSETS_DIR

with open("dataset_info.yaml") as f:
    ds_info = pd.DataFrame(yaml.safe_load(f))

ds_stats = []

dm_conf = hydra.utils.instantiate(OmegaConf.load("config/data/mode/mixed.yaml"))

for ds in ds_info["dataset_name"].values:
    ds_conf = hydra.utils.instantiate(
        OmegaConf.load(f"config/data/datasets/{ds}.yaml")[ds]
    )
    dm = TabularDataModule(
        dataset_config=ds_conf,
        data_mode_config=dm_conf,
    )

    dm.load()
    ds_stats.append(
        {
            "dataset_name": ds,
            "n_rows": dm.num_rows,
            "n_features": dm.num_features,
            "n_feature_cont": (~dm.feature_categ_mask).sum(),
            "n_feature_cat": dm.feature_categ_mask.sum(),
            "class_0": (dm.y == 0).sum(),
            "class_1": (dm.y == 1).sum(),
        }
    )

df = pd.DataFrame(ds_stats)
df = ds_info.merge(df, on="dataset_name")
df = df.sort_values("dataset_original_name", ascending=False)

fig = make_subplots(
    cols=2,
    subplot_titles=[
        "Label Ratio",
        "Feature Ratio",
    ],
    horizontal_spacing=0.05,
    vertical_spacing=0.07,
    shared_yaxes=True,
)

fig.add_trace(
    go.Bar(
        y=df["dataset_original_name"],
        x=df["n_feature_cont"],
        name="Continuous Features",
        orientation="h",
    ),
    row=1,
    col=1,
)
fig.add_trace(
    go.Bar(
        y=df["dataset_original_name"],
        x=df["n_feature_cont"] + df["n_feature_cat"],
        name="Categorical Features",
        orientation="h",
    ),
    row=1,
    col=1,
)


# Add bar chart for class 0 vs class 1
fig.add_trace(
    go.Bar(
        y=df["dataset_original_name"],
        x=df["class_0"],
        name="Class: 0",
        orientation="h",
    ),
    row=1,
    col=2,
)
fig.add_trace(
    go.Bar(
        y=df["dataset_original_name"],
        x=df["class_0"] + df["class_1"],
        name="Class: 1",
        orientation="h",
    ),
    row=1,
    col=2,
)

fig.update_layout(
    barmode="stack",
    # title="Dataset Statistics",
    # title_x = 0.5,
    # height = 600,
    # width = 1000,
    margin=dict(
        l=0,
        r=0,
        t=0,
        b=0,
        # pad = 20,
    ),
    hoverlabel=dict(
        bgcolor="#ededed",
        font_size=12,
        namelength=-1,
    ),
    yaxis_title="Datasets",
    xaxis=dict(title="Number of Instances per Class"),
    xaxis2=dict(title="Feature Composition"),
)

fig.update_yaxes(ticksuffix="  ")

fig.write_html(DASHBOARD_ASSETS_DIR / "plots/dataset_stats_plot.html")

# convert the dataset name to clickables
df["dataset_name_clickable"] = df[["dataset_original_name", "openml_url"]].apply(
    lambda x: f'<a href="{x.iloc[1]}">{x.iloc[0]}</a>', axis=1
    # lambda x: f'<a href="{x[1]}">{x[0]}</a>', axis=1
)

df = df.sort_values("dataset_original_name")
with open(DASHBOARD_ASSETS_DIR / "tables/dataset_stats_table.json", "w") as f:
    # pretty_df =
    json.dump(
        {
            "columns": [
                "Dataset",
                "# Instances",
                "# Features",
                "# Continuous Features",
                "# Categorical Features",
                "# Class 0",
                "# Class 1",
            ],
            "values": df[
                [
                    "dataset_name_clickable",
                    "n_rows",
                    "n_features",
                    "n_feature_cont",
                    "n_feature_cat",
                    "class_0",
                    "class_1",
                ]
            ].values.tolist(),
        },
        f,
        indent=2,
    )
