import shutil
from pathlib import Path
import hydra

from tabdd.config.pipeline_config import load_pipeline_configs

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

import __main__

if hasattr(__main__, "__file__"):
    # is script mode
    conf_dir = "../../config"
else:
    # is interactive mode
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
        "distill/methods=[random_sample,kmeans]",
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
        "distill/methods=[random_sample,kmeans,agglo,kip,gm]",
        "classifier/models=[xgb,ft_transformer,resnet,mlp,logistic_regression,gaussian_nb,knn]",
        "encoder/models=[mlp,gnn,tf]",
        "classifier.train.tune_hyperopt=false",
        "classifier.train.results_dir=data_mode_switch",
    ],
)

mixed_onehot_config = hydra.compose(
    config_name="tune",
    overrides=[
        f"data/datasets=[{','.join(datasets)}]",
        "data.mode.parse_mode=mixed",
        "distill/methods=[random_sample,kmeans]",
        "+distill.common.post_data_mode_name=onehot",
        "classifier/models=[xgb,ft_transformer,resnet,mlp,logistic_regression,gaussian_nb,knn]",
        "encoder/models=[]",
        "classifier.train.tune_hyperopt=false",
        "classifier.train.results_dir=data_mode_switch",
    ],
)

onehot_mixed_config = hydra.compose(
    config_name="tune",
    overrides=[
        f"data/datasets=[{','.join(datasets)}]",
        "data.mode.parse_mode=onehot",
        "distill/methods=[random_sample,kmeans,agglo,kip,gm]",
        "+distill.common.post_data_mode_name=mixed",
        "classifier/models=[xgb,ft_transformer,resnet,mlp,logistic_regression,gaussian_nb,knn]",
        "encoder/models=[mlp,gnn,tf]",
        "classifier.train.tune_hyperopt=false",
        "classifier.train.results_dir=data_mode_switch",
    ],
)


def update_results(config):
    data_mode_config = load_data_mode_config(config)
    dataset_configs = load_dataset_configs(config)
    encoder_tune_configs = load_encoder_tune_configs(config)
    encoder_train_config = load_encoder_train_config(config)
    distill_configs = load_distill_configs(config)
    classifier_tune_configs = load_classifier_tune_configs(config)
    classifier_train_config = load_classifier_train_config(config)

    for i_d, dataset_config in enumerate(dataset_configs):
        for i_c, classifier_tune_config in enumerate(classifier_tune_configs):
            for i_dis, distill_config in enumerate(distill_configs):
                pipeline_configs = load_pipeline_configs(
                    dataset_config=dataset_config,
                    data_mode_config=data_mode_config,
                    distill_config=distill_config,
                    classifier_tune_config=classifier_tune_config,
                    classifier_train_config=classifier_train_config,
                    encoder_tune_configs=encoder_tune_configs,
                    encoder_train_config=encoder_train_config,
                )
                for i_p, pipeline_config in enumerate(pipeline_configs):
                    report_dir = Path(pipeline_config.report_dir)
                    old_report_dir = Path(str(report_dir).replace("data_mode_switch", "tune_classifier_results"))

                    if not report_dir.is_file() and old_report_dir.is_file():
                        print("Updated")
                        print(pipeline_config.pretty_name, pipeline_config.distill_size, pipeline_config.random_seed)
                        print()
                        shutil.copy(old_report_dir, report_dir)

update_results(onehot_config)
