import hydra
from typing import Literal, Callable
from omegaconf import DictConfig
import pandas as pd
import numpy as np
from tabdd.utils import progress_bar
from pathlib import Path
import json

from tabdd.config import (
    DatasetConfig,
    EncoderTuneConfig,
    EncoderTrainConfig,
    ClassifierTuneConfig,
    ClassifierTrainConfig,
    DistillConfig,
    DataModeConfig,
    load_data_mode_config,
    load_dataset_configs,
    load_encoder_train_config,
    load_encoder_tune_configs,
    load_distill_configs,
    load_classifier_tune_configs,
    load_classifier_train_config,
    load_pipeline_configs,
)
from tabdd.results.load import load_all_clf_perf, compute_rank
from tabdd.results.plot.matplotlib import make_boxplot




conf_dir = "config"


hydra.initialize(config_path=conf_dir, version_base=None)

datasets = [
    "adult",
    "amazon_employee_access",
    "bank_marketing",
    "cardio_disease",
    "credit",
    "credit_default",
    "diabetes",
    "electricity",
    "elevators",
    "higgs",
    "home_equity_credit",
    "house",
    "internet_usage",
    "jannis",
    "law_school_admissions",
    "magic_telescope",
    "medical_appointments",
    "mini_boo_ne",
    "numer_ai",
    "nursery",
    # "online_shoppers",
    "phishing_websites",
    "pol",
    "road_safety",
    "tencent_ctr_small",
    "two_d_planes",
]

mixed_config = hydra.compose(
    config_name="tune",
    overrides=[
        f"data/datasets=[{','.join(datasets)}]",
        "data.mode.parse_mode=mixed",
        "distill/methods=[random_sample,kmeans,agglo,kip,gm]",
        "classifier/models=[xgb,ft_transformer,resnet,mlp,logistic_regression,gaussian_nb,knn]",
        "encoder/models=[]",
        "classifier.train.tune_hyperopt=false",
        "classifier.train.results_dir=data_mode_switch",
    ],
)

onehot_config = hydra.compose(
    config_name="tune",
    overrides=[
        f"data/datasets=[{','.join(datasets)}]",
        "data.mode.parse_mode=onehot",
        "distill/methods=[original,random_sample,kmeans,agglo,kip,gm]",
        "classifier/models=[xgb,ft_transformer,resnet,mlp,logistic_regression,gaussian_nb,knn]",
        "encoder/models=[mlp,gnn,tf]",
        "classifier.train.tune_hyperopt=false",
        "classifier.train.results_dir=data_mode_switch",
    ],
)

hydra.core.global_hydra.GlobalHydra.instance().clear()

from tabdd.results.load.classifier_performance import load_clf_perf
from tabdd.config import load_pipeline_configs

config = onehot_config

data_mode_config = load_data_mode_config(config)
dataset_configs = load_dataset_configs(config)
encoder_tune_configs = load_encoder_tune_configs(config)
encoder_train_config = load_encoder_train_config(config)
distill_configs = load_distill_configs(config)
classifier_tune_configs = load_classifier_tune_configs(config)
classifier_train_config = load_classifier_train_config(config)

results = []
incomplete = []

for i_ds, dataset_config in enumerate(dataset_configs):
    res_per_ds = []
    for i_c, classifier_tune_config in enumerate(classifier_tune_configs):
        for i_e, encoder_tune_config in enumerate(encoder_tune_configs):
            for i_d, distill_config in enumerate(distill_configs):
                result, incomplete_runs = load_clf_perf(
                    dataset_config=dataset_config,
                    data_mode_config=data_mode_config,
                    classifier_tune_config=classifier_tune_config,
                    classifier_train_config=classifier_train_config,
                    distill_config=distill_config,
                    encoder_train_config=encoder_train_config,
                    encoder_tune_config=encoder_tune_config,
                    refresh=True,
                )

                res_per_ds.append(result)
                incomplete.append(incomplete_runs)
    break

res_per_ds = pd.concat(r for r in res_per_ds if len(r))

if res_per_ds.isna().any().any():
    print(f"### ISSUE ###")
    print(f" there are nans.. :{res_per_ds.isna().any()}")

if res_per_ds["Data Distill Time"].isna().any():
    print("distill time nans...")
    print(
        res_per_ds[res_per_ds["Data Distill Time"].isna()][
            ["Data Distill Time", "Data Mode"]
        ]
    )
    quit()

# drop duplicates that do not depend on a encoder
no_encoder = res_per_ds[res_per_ds["Encoder"] == ""].drop_duplicates()
uses_encoder = res_per_ds[res_per_ds["Encoder"] != ""]

to_concat = []
if len(no_encoder):
    to_concat.append(no_encoder)
if len(uses_encoder):
    to_concat.append(uses_encoder)

res_per_ds = pd.concat(to_concat)

# print("we concated separated by encoder")
# res_w_reg = compute_regret(res_per_ds)
# results.append(res_w_reg)

