
# """
# Example: Manually loading config
# Notes: This does not always behave the same as using hydra.main(...)

import hydra
hydra.initialize(config_path="config", version_base=None)

config = hydra.compose(
    config_name="tune_classifier",
    overrides=[
        "data/datasets=[amazon_employee_access]",
        "data.mode.parse_mode=mixed",
        "distill/methods=[kmeans]",
        "+distill.methods.kmeans.post_data_mode_name=onehot",
        "classifier/models=[xgb,mlp,logistic_regression,gaussian_nb,knn]",
        "~encoder/models",
        "classifier.train.tune_hyperopt=false",
        "classifier.train.results_dir=tune_classifier_results",
    ]
)
# """

# """
# Example: Load invidivual component configs

from tabdd.config import (
    load_dataset_configs,
    load_data_mode_config,
    load_distill_configs,
    load_classifier_tune_configs,
    load_classifier_train_config,
    load_encoder_tune_configs,
    load_encoder_train_config,
)
data_mode_config = load_data_mode_config(config)
dataset_configs = load_dataset_configs(config)
encoder_tune_configs = load_encoder_tune_configs(config)
encoder_train_config = load_encoder_train_config(config)
distill_configs = load_distill_configs(config)
classifier_tune_configs = load_classifier_tune_configs(config)
classifier_train_config = load_classifier_train_config(config)
# """

# """
# Example: Loading pipeline configs

from tabdd.config import load_pipeline_configs
pipeline_configs = load_pipeline_configs(
    dataset_config = dataset_configs[0],
    data_mode_config = data_mode_config,
    classifier_tune_config = classifier_tune_configs[0],
    classifier_train_config = classifier_train_config,
    distill_config = distill_configs[0],
    encoder_train_config = encoder_train_config,
    encoder_tune_configs = encoder_tune_configs,
)
# """

# """
# Example: Loading the datamodule

from tabdd.data import TabularDataModule
dm = TabularDataModule(
    dataset_config=dataset_configs[0],
    data_mode_config=data_mode_config,
)
dm.load()
# """

"""
# Example: Loading the encoder

from tabdd.models.encoder import load_encoder
enc = load_encoder(
    encoder_tune_config=encoder_tune_configs[0],
    encoder_train_config=encoder_train_config,
    data_module = dm,
    device="cuda",
)
"""


# """
# Example: Loading the results

from tabdd.results.load import load_all_clf_perf
report, incomplete = load_all_clf_perf(config)
# """
