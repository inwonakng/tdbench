import hydra

from tabdd.config.paths import ROOT_DIR
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
from tabdd.utils import progress_bar

base_scripts_dir = ROOT_DIR / "scripts"
base_scripts_dir.mkdir(exist_ok=True, parents=True)

import __main__

if hasattr(__main__, "__file__"):
    # is script mode
    conf_dir = "../../config"
else:
    # is interactive mode
    conf_dir = "config"
hydra.initialize(config_path=conf_dir, version_base=None)

HAS_NO_CONT = ["nursery", "phishing_websites"]

######################
### CONFIGURE HERE ###
######################

RUN_NAME = "data-mode-switch"
DATASETS = [
    "adult",
    "amazon_employee_access",
    "bank_marketing",
    "credit",
    "credit_default",
    "diabetes",
    "electricity",
    "elevators",
    "higgs",
    "home_equity_credit",
    "house",
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
DATA_MODES = [
    "onehot",
    # "onehot-mixed",
    "mixed",
    # "mixed-onehot",
]
CLASSIFIERS = [
    "xgb",
    "ft_transformer",
    "resnet",
    "mlp",
    "logistic_regression",
    "gaussian_nb",
    "knn",
]
DISTILL_METHODS = [
    "original",
    # "encoded",
    # "decoded",
    "random_sample",
    "agglo",
    "kmeans",
    "kip",
    "gm",
]
ENCODERS = ["mlp", "gnn", "tf"]
ENCODER_TRAIN = "npl"
ENCODER_TRAIN_TARGETS = ["base", "multihead"]
LATENT_DIM = 16
RESULTS_DIR = "data_mode_switch"
CHECKPOINT_DIR = "best_checkpoints"
TUNE_HYPEROPT = "false"

######################
### CONFIGURE DONE ###
######################


def parse_data_mode(data_mode: str):
    if data_mode == "onehot":
        return ["data.mode.parse_mode=onehot"]
    elif data_mode == "onehot-mixed":
        return [
            "data.mode.parse_mode=onehot",
            "+distill.common.post_data_mode_name=mixed",
        ]
    elif data_mode == "mixed":
        return ["data.mode.parse_mode=mixed"]
    elif data_mode == "mixed-onehot":
        return [
            "data.mode.parse_mode=mixed",
            "+distill.common.post_data_mode_name=onehot",
        ]
    else:
        raise ValueError(f"Unknown data mode: {data_mode}")


def hydrafy(options):
    return '"[' + ",".join(options) + ']"'


def check_if_done(overrides):
    config = hydra.compose(
        config_name="tune",
        overrides=[o.replace('"', "") for o in overrides],
    )
    data_mode_config = load_data_mode_config(config)
    dataset_configs = load_dataset_configs(config)
    encoder_tune_configs = load_encoder_tune_configs(config)
    encoder_train_config = load_encoder_train_config(config)
    distill_configs = load_distill_configs(config)
    classifier_tune_configs = load_classifier_tune_configs(config)
    classifier_train_config = load_classifier_train_config(config)

    return not any(
        not p.is_complete
        for dataset_config in dataset_configs
        for distill_config in distill_configs
        for classifier_tune_config in classifier_tune_configs
        for p in load_pipeline_configs(
            dataset_config=dataset_config,
            data_mode_config=data_mode_config,
            distill_config=distill_config,
            classifier_tune_config=classifier_tune_config,
            classifier_train_config=classifier_train_config,
            encoder_tune_configs=encoder_tune_configs,
            encoder_train_config=encoder_train_config,
        )
    )


def get_pipeline_count(overrides):
    config = hydra.compose(
        config_name="tune",
        overrides=[o.replace('"', "") for o in overrides],
    )
    data_mode_config = load_data_mode_config(config)
    dataset_configs = load_dataset_configs(config)
    encoder_tune_configs = load_encoder_tune_configs(config)
    encoder_train_config = load_encoder_train_config(config)
    distill_configs = load_distill_configs(config)
    classifier_tune_configs = load_classifier_tune_configs(config)
    classifier_train_config = load_classifier_train_config(config)

    return len(
        [
            p
            for dataset_config in dataset_configs
            for distill_config in distill_configs
            for classifier_tune_config in classifier_tune_configs
            for p in load_pipeline_configs(
                dataset_config=dataset_config,
                data_mode_config=data_mode_config,
                distill_config=distill_config,
                classifier_tune_config=classifier_tune_config,
                classifier_train_config=classifier_train_config,
                encoder_tune_configs=encoder_tune_configs,
                encoder_train_config=encoder_train_config,
            )
        ]
    )


with progress_bar() as progress:
    task = progress.add_task("Counting pipelines..", total=len(DATASETS) * len(DATA_MODES) * len(CLASSIFIERS) * len(DISTILL_METHODS))
    pipeline_count = 0
    for ds in DATASETS:
        for dm in DATA_MODES:
            if ds in ["nursery", "phishing_websites"] and dm in [
                "onehot-mixed",
                "mixed-onehot",
            ]:
                continue
            for clf in CLASSIFIERS:
                tasks = []
                for dd in DISTILL_METHODS:
                    jobname = f"{ds}-{dm}-{clf}"

                    task_name = f"{jobname}-{dd}"

                    task_args = [
                        f"data/datasets={hydrafy([ds])}",
                        f"distill/methods={hydrafy([dd])}",
                        f"classifier/models={hydrafy([clf])}",
                        f"classifier.train.results_dir={RESULTS_DIR}",
                        f"classifier.train.tune_hyperopt={TUNE_HYPEROPT}",
                        f"encoder/train={ENCODER_TRAIN}",
                        f"encoder.train.latent_dim={LATENT_DIM}",
                        f"encoder.train.checkpoint_dir={CHECKPOINT_DIR}",
                        f"encoder.train.train_target={hydrafy(ENCODER_TRAIN_TARGETS)}",
                    ]

                    if dm in ["mixed", "mixed-onehot"]:
                        task_args.append('encoder/models="[]"')
                    else:
                        task_args.append(f"encoder/models={hydrafy(ENCODERS)}")

                    task_args += parse_data_mode(dm)

                    # comment out if complete, but stil keep it there for debugging purposes..
                    pipeline_count += get_pipeline_count(task_args)
                    progress.update(task, advance=1)
    print(f"total pipelines count: {pipeline_count}")
