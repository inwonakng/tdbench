import hydra

from tabdd.data import TabularDataModule
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


mixed_config = hydra.compose(
    config_name="tune_classifier",
    overrides=[
        "data/datasets=[amazon_employee_access]",
        "data.mode.parse_mode=onehot",
        "distill/methods=[original,random_sample,kmeans]",
        "classifier/models=[xgb,mlp,logistic_regression,gaussian_nb,knn]",
        "encoder/models=[mlp,gnn,tf]",
        "classifier.train.tune_hyperopt=false",
        "classifier.train.results_dir=data_mode_switch",
    ],
)

mixed_config = hydra.compose(
    config_name="tune_classifier",
    overrides=[
        "data/datasets=[amazon_employee_access]",
        "data.mode.parse_mode=mixed",
        "distill/methods=[kmeans]",
        "+distill.methods.kmeans.post_data_mode_name=onehot",
        "classifier/models=[xgb]",
        "~encoder/models",
        "classifier.train.tune_hyperopt=false",
        "classifier.train.results_dir=tune_classifier_results",
    ],
)

onehot_config = hydra.compose(
    config_name="tune_classifier",
    overrides=[
        "data/datasets=[amazon_employee_access]",
        "data.mode.parse_mode=onehot",
        "distill/methods=[kmeans]",
        "+distill.methods.kmeans.post_data_mode_name=mixed",
        "classifier/models=[xgb]",
        "~encoder/models",
        "classifier.train.tune_hyperopt=false",
        "classifier.train.results_dir=tune_classifier_results",
    ],
)

pipeline_configs = load_pipeline_configs(
    dataset_config = load_dataset_configs(mixed_config)[0],
    data_mode_config = load_data_mode_config(mixed_config),
    classifier_tune_config = load_classifier_tune_configs(mixed_config)[0],
    classifier_train_config = load_classifier_train_config(mixed_config),
    distill_config = load_distill_configs(mixed_config)[0],
    encoder_tune_configs = load_encoder_tune_configs(mixed_config),
    encoder_train_config = load_encoder_train_config(mixed_config),
)

pipeline_configs += load_pipeline_configs(
    dataset_config = load_dataset_configs(onehot_config)[0],
    data_mode_config = load_data_mode_config(onehot_config),
    classifier_tune_config = load_classifier_tune_configs(onehot_config)[0],
    classifier_train_config = load_classifier_train_config(onehot_config),
    distill_config = load_distill_configs(onehot_config)[0],
    encoder_tune_configs = load_encoder_tune_configs(onehot_config),
    encoder_train_config = load_encoder_train_config(onehot_config),
)

mixed_centroid_to_onehot = None
mixed_closest_to_onehot = None
onehot_centroid_to_mixed = None
onehot_closest_to_mixed = None

for pconf in pipeline_configs:
    if pconf.cluster_center == "centroid":
        if (
            pconf.data_mode_config.parse_mode == "onehot"
            and pconf.distill_config.post_data_mode_config.parse_mode == "mixed"
        ):
            onehot_centroid_to_mixed = pconf
        elif (
            pconf.data_mode_config.parse_mode == "mixed"
            and pconf.distill_config.post_data_mode_config.parse_mode
            == "onehot"
        ):
            mixed_centroid_to_onehot = pconf
    if pconf.cluster_center == "closest":
        if (
            pconf.data_mode_config.parse_mode == "onehot"
            and pconf.distill_config.post_data_mode_config.parse_mode == "mixed"
        ):
            onehot_closest_to_mixed = pconf
        elif (
            pconf.data_mode_config.parse_mode == "mixed"
            and pconf.distill_config.post_data_mode_config.parse_mode
            == "onehot"
        ):
            mixed_closest_to_onehot = pconf

hydra.core.global_hydra.GlobalHydra.instance().clear()

def test_onehot_centroid_to_mixed():
    dm = TabularDataModule(
        dataset_config=onehot_centroid_to_mixed.dataset_config,
        data_mode_config=onehot_centroid_to_mixed.distill_config.post_data_mode_config,
    )
    dm.load()
    distilled = load_distilled_data(onehot_centroid_to_mixed, use_cache=False)
    assert distilled["train"].X.shape[1] == dm.X.shape[1]

def test_mixed_centroid_to_onehot():
    dm = TabularDataModule(
        dataset_config=mixed_centroid_to_onehot.dataset_config,
        data_mode_config=mixed_centroid_to_onehot.distill_config.post_data_mode_config,
    )
    dm.load()
    distilled = load_distilled_data(mixed_centroid_to_onehot, use_cache=False)
    assert distilled["train"].X.shape[1] == dm.X.shape[1]

def test_onehot_closest_to_mixed():
    dm = TabularDataModule(
        dataset_config=onehot_closest_to_mixed.dataset_config,
        data_mode_config=onehot_closest_to_mixed.distill_config.post_data_mode_config,
    )
    dm.load()
    print(onehot_closest_to_mixed.data_mode_config.identifier)
    print(onehot_closest_to_mixed.distill_config.post_data_mode_config.identifier)
    distilled = load_distilled_data(onehot_closest_to_mixed, use_cache=False)
    assert distilled["train"].X.shape[1] == dm.X.shape[1]

def test_mixed_closest_to_onehot():
    dm = TabularDataModule(
        dataset_config=mixed_closest_to_onehot.dataset_config,
        data_mode_config=mixed_closest_to_onehot.distill_config.post_data_mode_config,
    )
    dm.load()
    distilled = load_distilled_data(mixed_closest_to_onehot, use_cache=False)
    assert distilled["train"].X.shape[1] == dm.X.shape[1]

def test_onehot_distill():
    onehot_dm = TabularDataModule(
        dataset_config=mixed_closest_to_onehot.dataset_config,
        data_mode_config=mixed_closest_to_onehot.distill_config.post_data_mode_config,
    )
    onehot_dm.load()
    to_onehot = load_distilled_data(mixed_closest_to_onehot, use_cache=False)

    assert to_onehot["train"].feature_mask.shape[0] == to_onehot["train"].X.shape[1]
    assert (to_onehot["train"].feature_categ_mask == onehot_dm.feature_categ_mask).all()
    assert to_onehot["train - original"].feature_mask.shape[0] == to_onehot["train - original"].X.shape[1]
    assert (to_onehot["train - original"].feature_categ_mask == onehot_dm.feature_categ_mask).all()
    assert to_onehot["val"].feature_mask.shape[0] == to_onehot["val"].X.shape[1]
    assert (to_onehot["val"].feature_categ_mask == onehot_dm.feature_categ_mask).all()
    assert to_onehot["test"].feature_mask.shape[0] == to_onehot["test"].X.shape[1]
    assert (to_onehot["test"].feature_categ_mask == onehot_dm.feature_categ_mask).all()

def test_mixed_distill():
    mixed_dm = TabularDataModule(
        dataset_config=onehot_closest_to_mixed.dataset_config,
        data_mode_config=onehot_closest_to_mixed.distill_config.post_data_mode_config,
    )
    mixed_dm.load()
    to_mixed = load_distilled_data(onehot_closest_to_mixed, use_cache=False)

    assert to_mixed["train"].feature_mask.shape[0] == to_mixed["train"].X.shape[1]
    assert (to_mixed["train"].feature_categ_mask == mixed_dm.feature_categ_mask).all()
    assert to_mixed["train - original"].feature_mask.shape[0] == to_mixed["train - original"].X.shape[1]
    assert (to_mixed["train - original"].feature_categ_mask == mixed_dm.feature_categ_mask).all()
    assert to_mixed["val"].feature_mask.shape[0] == to_mixed["val"].X.shape[1]
    assert (to_mixed["val"].feature_categ_mask == mixed_dm.feature_categ_mask).all()
    assert to_mixed["test"].feature_mask.shape[0] == to_mixed["test"].X.shape[1]
    assert (to_mixed["test"].feature_categ_mask == mixed_dm.feature_categ_mask).all()
