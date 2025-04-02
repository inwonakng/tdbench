import hydra
import pandas as pd
import numpy as np
import torch
from torch import nn, optim
import copy
import itertools
from torch.utils.tensorboard import SummaryWriter
import json
import matplotlib.pyplot as plt
import seaborn as sns

from tabdd.config import (
    load_dataset_configs,
    load_data_mode_config,
    load_distill_configs,
    load_classifier_tune_configs,
    load_classifier_train_config,
    load_encoder_tune_configs,
    load_encoder_train_config,
    load_pipeline_configs,
)
from tabdd.config.paths import RESULTS_CACHE_DIR
from tabdd.results.load import load_all_clf_perf, compute_rank, load_all_enc_stats
from tabdd.results.plot.matplotlib import radar_factory, make_radarplot
from tabdd.distill import load_distilled_data
from tabdd.data import TabularDataModule

from tabdd.results.plot.ranking_radar import get_dm_ranks_per_enc


hydra.initialize(config_path="config", version_base=None)

config = hydra.compose(
    config_name="all",
    overrides=[
        "data/datasets=[amazon_employee_access]",
        "classifier.train.results_dir=runtime_clf_results",
        "classifier.train.tune_hyperopt=true",
        "classifier/models=[xgb]",
        "encoder/models=[mlp]",
        "encoder.train.checkpoint_dir=runtime_checkpoints",
        "distill/methods=[original,encoded,decoded,random_sample,agglo,kip,kmeans,gm]",
        "refresh_results=true",
    ],
)

report, incomplete = load_all_clf_perf(config)
distill_methods = list(
    set(report["Distill Method"].unique())
    - set(["Original", "Mixed Original", "Encoded", "Decoded"])
)
encoders = list(set(report["Encoder"].unique()) - set([""]))

overall_ranks = get_dm_ranks_per_enc(report, "all")
no_enc_ranks = get_dm_ranks_per_enc(report, "")

radar_factory(len(overall_ranks))
fig, axes = plt.subplots(
    nrows=2, ncols=4, subplot_kw=dict(projection="radar"), figsize=(16, 7)
)
fig.subplots_adjust(wspace=0.4, hspace=0.20, top=1.2, bottom=0.05)

data=overall_ranks.values[:, 1:]
variables=overall_ranks.values[:, 0]
ax=axes[0, 0]
frame="polygon"
colors = None

theta = radar_factory(len(variables), frame=frame)
if colors is None:
    colors = ["r", "b", "g", "m", "y", "c", "k"]
for row, c in zip(data.T, colors):
    ax.plot(theta, row, color=c)
    ax.fill(theta, row, facecolor=c, alpha=0.25, label="_nolegend_")


make_radarplot(
    data=overall_ranks.values[:, 1:],
    variables=overall_ranks.values[:, 0],
    ax=axes[0, 0],
)
axes[0,0].set_title("Overall")

make_radarplot(
    data=no_enc_ranks.values[:, 1:],
    variables=no_enc_ranks.values[:, 0],
    ax=axes[1, 0],
)
axes[1,0].set_title("No Encoder")


enc_stats = load_all_enc_stats(config)


for crit in ["Total Tune Runtime", "Encoder Params", "Test Predict Accuracy"]:
    vals = enc_stats[crit].argsort()
    break

    tr_runtime_rank = zip(
        enc_stats["Model"],
        ["Tune Runtime"] * len(enc_stats),
        enc_stats["Total Tune Runtime"].argsort()
    )






distill_sizes = list(
    range(
        config.distill.common.distill_size_lower,
        config.distill.common.distill_size_upper + 1,
        config.distill.common.distill_size_step,
    )
)

valid_settings = [
    "Agglo -> Original / Centroid",
    # 'Agglo -> Original / Closest',
    # 'Agglo-MLP-Encoded -> Decoded / Centroid',
    # 'Agglo-MLP-Encoded -> Decoded / Closest',
    # 'Agglo-MLP-Encoded -> Decoded-Binary / Centroid',
    # 'Agglo-MLP-Encoded -> Decoded-Binary / Closest',
    # 'Agglo-MLP-Encoded -> Encoded / Centroid',
    # 'Agglo-MLP-Encoded -> Encoded / Closest',
    "Agglo-MLP-MultiHead-Encoded -> Decoded / Centroid",
    # 'Agglo-MLP-MultiHead-Encoded -> Decoded / Closest',
    "Agglo-MLP-MultiHead-Encoded -> Decoded-Binary / Centroid",
    # 'Agglo-MLP-MultiHead-Encoded -> Decoded-Binary / Closest',
    "Agglo-MLP-MultiHead-Encoded -> Encoded / Centroid",
    # 'Agglo-MLP-MultiHead-Encoded -> Encoded / Closest',
    "KIP -> Original / Centroid",
    # 'KIP -> Original / Closest',
    # 'KIP-MLP-Encoded -> Decoded / Centroid',
    # 'KIP-MLP-Encoded -> Decoded / Closest',
    # 'KIP-MLP-Encoded -> Decoded-Binary / Centroid',
    # 'KIP-MLP-Encoded -> Decoded-Binary / Closest',
    # 'KIP-MLP-Encoded -> Encoded / Centroid',
    # 'KIP-MLP-Encoded -> Encoded / Closest',
    "KIP-MLP-MultiHead-Encoded -> Decoded / Centroid",
    # 'KIP-MLP-MultiHead-Encoded -> Decoded / Closest',
    "KIP-MLP-MultiHead-Encoded -> Decoded-Binary / Centroid",
    # 'KIP-MLP-MultiHead-Encoded -> Decoded-Binary / Closest',
    "KIP-MLP-MultiHead-Encoded -> Encoded / Centroid",
    # 'KIP-MLP-MultiHead-Encoded -> Encoded / Closest',
    "KMeans -> Original / Centroid",
    # 'KMeans -> Original / Closest',
    # 'KMeans-MLP-Encoded -> Decoded / Centroid',
    # 'KMeans-MLP-Encoded -> Decoded / Closest',
    # 'KMeans-MLP-Encoded -> Decoded-Binary / Centroid',
    # 'KMeans-MLP-Encoded -> Decoded-Binary / Closest',
    # 'KMeans-MLP-Encoded -> Encoded / Centroid',
    # 'KMeans-MLP-Encoded -> Encoded / Closest',
    "KMeans-MLP-MultiHead-Encoded -> Decoded / Centroid",
    # 'KMeans-MLP-MultiHead-Encoded -> Decoded / Closest',
    "KMeans-MLP-MultiHead-Encoded -> Decoded-Binary / Centroid",
    # 'KMeans-MLP-MultiHead-Encoded -> Decoded-Binary / Closest',
    "KMeans-MLP-MultiHead-Encoded -> Encoded / Centroid",
    # 'KMeans-MLP-MultiHead-Encoded -> Encoded / Closest',
    # 'MLP Decoded',
    "MLP-MultiHead Decoded",
    "Mixed Original",
    "Original",
    # 'MLP Encoded',
    "MLP-MultiHead Encoded",
    "Mixed Random Sample",
    "Random Sample",
]

all_done = []
for dm, rows in report.groupby("Data Mode"):
    if len(rows["N"].unique()) == len(distill_sizes):
        all_done.append(rows)
all_done = pd.concat(all_done)
all_done = all_done[all_done["Subset"] == "Test"]

all_done["Perf/Time Gain"] = all_done["Score"] / all_done["Opt Train Time Total"]

fig, axes = plt.subplots(ncols=3)
sns.lineplot(
    data=all_done[all_done["Data Mode"].isin(valid_settings)],
    x="N",
    y="Score",
    ax=axes[0],
    hue="Short Name",
).set(title="Score over distillation size")

sns.lineplot(
    data=all_done[all_done["Data Mode"].isin(valid_settings)],
    x="N",
    y="Opt Train Time Total",
    ax=axes[1],
    hue="Short Name",
).set(
    title="Runtime over distillation size",
    yscale="log",
)

sns.lineplot(
    data=all_done[all_done["Data Mode"].isin(valid_settings)],
    x="N",
    y="Perf/Time Gain",
    ax=axes[2],
    hue="Short Name",
).set(
    title="Perf/Time Gain over distillation size",
    # yscale="log",
)
