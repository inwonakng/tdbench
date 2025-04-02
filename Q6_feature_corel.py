# %%
# first pick a dataset to read

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

import hydra
from tabdd.config import (
    load_data_mode_config,
    load_dataset_configs,
    load_encoder_tune_configs,
    load_encoder_train_config,
    load_distill_configs,
    load_classifier_tune_configs,
    load_classifier_train_config,
    load_pipeline_configs,
)
from tabdd.distill import load_distilled_data

# %%

"""
Candidate datasets:

- credit_default
- credit
- magic_telescope
- tencent_ctr_small
- two_d_planes

"""

dataset = "tencent_ctr_small"

hydra.initialize(config_path="config", version_base=None)
ori_config = hydra.compose(
    config_name="tune",
    overrides=[
        f"data/datasets=[{dataset}]",
        "data.mode.parse_mode=mixed",
        "distill/methods=[original]",
        "distill/common=n_100",
        "classifier/models=[xgb]",
        "classifier.train.results_dir=feature_corr",
    ],
)

ori_conf = load_pipeline_configs(
    dataset_config=load_dataset_configs(ori_config)[0],
    data_mode_config=load_data_mode_config(ori_config),
    classifier_tune_config=load_classifier_tune_configs(ori_config)[0],
    classifier_train_config=load_classifier_train_config(ori_config),
    distill_config=load_distill_configs(ori_config)[0],
    encoder_train_config=load_encoder_train_config(ori_config),
    encoder_tune_configs=load_encoder_tune_configs(ori_config),
)

rnd_config = hydra.compose(
    config_name="tune",
    overrides=[
        f"data/datasets=[{dataset}]",
        "data.mode.parse_mode=mixed",
        "distill/methods=[random_sample]",
        "classifier/models=[xgb]",
        "classifier.train.results_dir=feature_corr",
    ],
)

rnd_confs = load_pipeline_configs(
    dataset_config=load_dataset_configs(rnd_config)[0],
    data_mode_config=load_data_mode_config(rnd_config),
    classifier_tune_config=load_classifier_tune_configs(rnd_config)[0],
    classifier_train_config=load_classifier_train_config(rnd_config),
    distill_config=load_distill_configs(rnd_config)[0],
    encoder_train_config=load_encoder_train_config(rnd_config),
    encoder_tune_configs=load_encoder_tune_configs(rnd_config),
)

km_config = hydra.compose(
    config_name="tune",
    overrides=[
        f"data/datasets=[{dataset}]",
        "data.mode.parse_mode=onehot",
        "distill/methods=[kmeans]",
        "+distill.common.post_data_mode_name=mixed",
        "classifier/models=[xgb]",
        "encoder/models=[tf]",
        "classifier.train.results_dir=feature_corr",
        "encoder.train.train_target=[multihead]",
    ],
)

km_confs = load_pipeline_configs(
    dataset_config=load_dataset_configs(km_config)[0],
    data_mode_config=load_data_mode_config(km_config),
    classifier_tune_config=load_classifier_tune_configs(km_config)[0],
    classifier_train_config=load_classifier_train_config(km_config),
    distill_config=load_distill_configs(km_config)[0],
    encoder_train_config=load_encoder_train_config(km_config),
    encoder_tune_configs=load_encoder_tune_configs(km_config),
)

gm_config = hydra.compose(
    config_name="tune",
    overrides=[
        f"data/datasets=[{dataset}]",
        "data.mode.parse_mode=onehot",
        "distill/methods=[gm]",
        "+distill.common.post_data_mode_name=mixed",
        "classifier/models=[xgb]",
        "encoder/models=[tf]",
        "classifier.train.results_dir=feature_corr",
        "encoder.train.train_target=[multihead]",
    ],
)

gm_confs = load_pipeline_configs(
    dataset_config=load_dataset_configs(gm_config)[0],
    data_mode_config=load_data_mode_config(gm_config),
    classifier_tune_config=load_classifier_tune_configs(gm_config)[0],
    classifier_train_config=load_classifier_train_config(gm_config),
    distill_config=load_distill_configs(gm_config)[0],
    encoder_train_config=load_encoder_train_config(gm_config),
    encoder_tune_configs=load_encoder_tune_configs(gm_config),
)


hydra.core.global_hydra.GlobalHydra.instance().clear()

ori = ori_conf[0]

rnd = None
for cc in rnd_confs:
    attr = cc.attributes
    if attr["N"] == 100 and cc.random_seed == 0:
        rnd = cc

# pick out the config that is the original and enc+distill+dec
km_dec = None
km_ori = None
for cc in km_confs:
    attr = cc.attributes
    if (
        attr["Data Parse Mode"] == "onehot"
        and attr["Post Data Parse Mode"] == "mixed"
        and attr["Distill Space"] == "encoded"
        and attr["N"] == 100
        and cc.random_seed == 0
    ):
        if (
            attr["Output Space"] == "decoded"
            and attr["Cluster Center"] == "centroid"
            and not attr["Convert Binary"]
        ):
            km_dec = cc
        if attr["Output Space"] == "original" and attr["Cluster Center"] == "closest":
            km_ori = cc

# pick out the config that is the original and enc+distill+dec
gm = None
for cc in gm_confs:
    attr = cc.attributes
    if (
        attr["Data Parse Mode"] == "onehot"
        and attr["Post Data Parse Mode"] == "mixed"
        and attr["Distill Space"] == "encoded"
        and attr["N"] == 100
        and cc.random_seed == 0
    ):
        if (
            attr["Output Space"] == "decoded"
            and attr["Cluster Center"] == "centroid"
            and attr["Convert Binary"]
        ):
            gm = cc

# load the datas
ori_data = load_distilled_data(ori)
rnd_data = load_distilled_data(rnd)
km_dec_data = load_distilled_data(km_dec)
km_ori_data = load_distilled_data(km_ori)
gm_data = load_distilled_data(gm)

ori_x, ori_y = ori_data["train"].X, ori_data["train"].y
rnd_x, rnd_y = rnd_data["train"].X, rnd_data["train"].y
km_dec_x, km_dec_y = km_dec_data["train"].X, km_dec_data["train"].y
km_ori_x, km_ori_y = km_ori_data["train"].X, km_ori_data["train"].y
gm_x, gm_y = gm_data["train"].X, gm_data["train"].y

# fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(8, 2.5))
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(5, 2.5))

sns.heatmap(
    np.corrcoef(ori_x.T),
    annot=False,
    cmap="coolwarm",
    vmin=-1,
    vmax=1,
    center=0,
    ax=axes[0],
)
axes[0].set_title("Original")

# sns.heatmap(
#     np.corrcoef(rnd_x.T),
#     annot=False,
#     cmap="coolwarm",
#     vmin=-1,
#     vmax=1,
#     center=0,
#     ax=axes[1],
# )
# axes[1].set_title("Random Sample")

sns.heatmap(
    np.corrcoef(km_ori_x.T),
    annot=False,
    cmap="coolwarm",
    vmin=-1,
    vmax=1,
    center=0,
    ax=axes[1],
)
axes[1].set_title("$k$-means")

fig.tight_layout()
fig.savefig(
    f"./iclr-rebuttal-figures/rq6-feature-corr-{dataset}.pdf", bbox_inches="tight"
)
