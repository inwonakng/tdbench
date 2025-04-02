import hydra
import pytest
from sklearn.metrics import get_scorer

from tabdd.config import load_pipeline_configs
from tabdd.config import (
    load_dataset_configs,
    load_data_mode_config,
    load_distill_configs,
    load_classifier_tune_configs,
    load_classifier_train_config,
    load_encoder_tune_configs,
    load_encoder_train_config,
)
from tabdd.distill import load_distilled_data

hydra.initialize(config_path="../../config", version_base=None)

onehot_config = hydra.compose(
    config_name="tune",
    overrides=[
        "data/datasets=[pol]",
        "data.mode.parse_mode=onehot",
        "distill/methods=[kmeans]",
        "classifier/models=[ft_transformer]",
        "encoder/models=[mlp]",
        "classifier.train.tune_hyperopt=false",
        "classifier.train.results_dir=tune_classifier_results",
    ],
)

mixed_config = hydra.compose(
    config_name="tune",
    overrides=[
        "data/datasets=[pol]",
        "data.mode.parse_mode=mixed",
        "distill/methods=[kmeans]",
        "classifier/models=[ft_transformer]",
        "encoder/models=[]",
        "classifier.train.tune_hyperopt=false",
        "classifier.train.results_dir=tune_classifier_results",
    ],
)

onehot_pipeline_configs = load_pipeline_configs(
    dataset_config=load_dataset_configs(onehot_config)[0],
    data_mode_config=load_data_mode_config(onehot_config),
    classifier_tune_config=load_classifier_tune_configs(onehot_config)[0],
    classifier_train_config=load_classifier_train_config(onehot_config),
    distill_config=load_distill_configs(onehot_config)[0],
    encoder_tune_configs=load_encoder_tune_configs(onehot_config),
    encoder_train_config=load_encoder_train_config(onehot_config),
)

mixed_pipeline_configs = load_pipeline_configs(
    dataset_config=load_dataset_configs(mixed_config)[0],
    data_mode_config=load_data_mode_config(mixed_config),
    classifier_tune_config=load_classifier_tune_configs(mixed_config)[0],
    classifier_train_config=load_classifier_train_config(mixed_config),
    distill_config=load_distill_configs(mixed_config)[0],
    encoder_tune_configs=load_encoder_tune_configs(mixed_config),
    encoder_train_config=load_encoder_train_config(mixed_config),
)


hydra.core.global_hydra.GlobalHydra.instance().clear()

km_oh_cent = None
km_oh_cls = None
km_oh_enc = None
km_mi_cent = None
km_mi_cls = None

for p in onehot_pipeline_configs:
    if p.distill_method_name == "kmeans":
        if p.cluster_center == "centroid" and not km_oh_cent:
            km_oh_cent = p
        if p.cluster_center == "closest" and not km_oh_cls:
            km_oh_cls = p
        if (
            p.encoder_pretty_name != "N/A"
            and p.output_space == "encoded"
            and not km_oh_enc
        ):
            km_oh_enc = p

for p in mixed_pipeline_configs:
    if p.distill_method_name == "kmeans":
        if p.cluster_center == "centroid" and not km_mi_cent:
            km_mi_cent = p
        if p.cluster_center == "closest" and not km_mi_cls:
            km_mi_cls = p


classifier_args = dict(
    d_block=16,
    n_blocks=2,
    attention_n_heads=2,
    attention_dropout=0.001,
    ffn_d_hidden_multiplier=2,
    ffn_dropout=0,
    residual_dropout=0,
    opt_name="Adam",
    opt_lr=0.01,
    opt_wd=0,
)


scorer = get_scorer("balanced_accuracy")


def train_and_fit(classifier, distilled_data):
    classifier.fit(distilled_data["train"].X, distilled_data["train"].y)
    scorer(classifier, distilled_data["train"].X, distilled_data["train"].y)
    classifier.fit(
        distilled_data["train - original"].X[:100],
        distilled_data["train - original"].y[:100],
    )
    scorer(
        classifier,
        distilled_data["train - original"].X[:100],
        distilled_data["train - original"].y[:100],
    )
    classifier.fit(distilled_data["val"].X, distilled_data["val"].y)
    scorer(classifier, distilled_data["val"].X, distilled_data["val"].y)
    classifier.fit(distilled_data["test"].X, distilled_data["test"].y)
    scorer(classifier, distilled_data["val"].X, distilled_data["val"].y)


def test_fit_km_oh_cent():
    distilled_data = load_distilled_data(km_oh_cent, use_cache=False)
    classifier = km_oh_cent.classifier_tune_config.instantiate(
        params=dict(
            **classifier_args,
            sample_dset=distilled_data["train - original"],
        )
    )
    train_and_fit(classifier, distilled_data)


def test_fit_km_oh_cls():
    distilled_data = load_distilled_data(km_oh_cls, use_cache=False)
    classifier = km_oh_cls.classifier_tune_config.instantiate(
        params=dict(
            **classifier_args,
            sample_dset=distilled_data["train - original"],
        )
    )
    train_and_fit(classifier, distilled_data)


def test_fit_km_oh_enc():
    distilled_data = load_distilled_data(km_oh_enc, use_cache=False)
    classifier = km_oh_enc.classifier_tune_config.instantiate(
        params=dict(
            **classifier_args,
            sample_dset=distilled_data["train - original"],
        )
    )
    train_and_fit(classifier, distilled_data)


def test_fit_km_mi_cent():
    distilled_data = load_distilled_data(km_mi_cent, use_cache=False)
    classifier = km_mi_cent.classifier_tune_config.instantiate(
        params=dict(
            **classifier_args,
            sample_dset=distilled_data["train - original"],
        )
    )
    train_and_fit(classifier, distilled_data)


def test_fit_km_mi_cls():
    distilled_data = load_distilled_data(km_mi_cls, use_cache=False)
    classifier = km_mi_cls.classifier_tune_config.instantiate(
        params=dict(
            **classifier_args,
            sample_dset=distilled_data["train - original"],
        )
    )

    train_and_fit(classifier, distilled_data)
