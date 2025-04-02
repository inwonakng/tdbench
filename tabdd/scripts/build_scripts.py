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
    # "original",
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


scripts_dir = base_scripts_dir / RUN_NAME
batch_script_dir = scripts_dir / "batch"
batch_script_dir.mkdir(exist_ok=True, parents=True)
task_script_dir = scripts_dir / "task"
task_script_dir.mkdir(exist_ok=True, parents=True)
group_script_dir = scripts_dir / "task_group"
group_script_dir.mkdir(exist_ok=True, parents=True)

with open(base_scripts_dir / "sample_batch.sh") as f:
    sample_batch_script = f.read()
with open(base_scripts_dir / "sample_script.sh") as f:
    sample_script = f.read()
with open(base_scripts_dir / "sample_group.sh") as f:
    sample_group = f.read()
conda_script = base_scripts_dir / "activate_conda.sh"

task_complete_count = 0
task_count = 0

group_complete_count = 0
group_count = 0

group_scripts = []

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

                logfile = f"scripts/{RUN_NAME}/logs/{jobname}.out"
                command = (
                    "python -u -m tabdd.tune.classifier \\\n"
                    + " \\\n".join("  " + a for a in task_args)
                    + " \\\n"
                )

                script = sample_script.replace("%%COMMAND%%", command).replace(
                    "%%TASK_NAME%%", task_name
                )

                with open(task_script_dir / f"{task_name}.sh", "w") as f:
                    f.write(script)
                print("Created task:", task_name)

                # comment out if complete, but stil keep it there for debugging purposes..
                task_is_done = check_if_done(task_args)
                prefix = "# " if task_is_done else ""
                task_complete_count += task_is_done
                task_count += 1

                tasks.append(
                    # f"{prefix}mpirun -n 1 bash scripts/{RUN_NAME}/task/{task_name}.sh &"
                    f"{prefix}bash scripts/{RUN_NAME}/task/{task_name}.sh &"
                )

            group_count += 1

            # if there are no more tasks that need to be ran, skip..
            # print([t.startswith("#") for t in tasks])
            if all(t.startswith("#") for t in tasks):
                print("-" * 40)
                print("Skipping completed task group:", jobname)
                print("-" * 40)
                group_complete_count += 1
                continue

            with open(group_script_dir / f"{jobname}.sh", "w") as f:
                f.write(sample_group.replace("%%TASKS%%", "\n".join(tasks)))
            with open(batch_script_dir / f"{jobname}.sh", "w") as f:
                f.write(
                    sample_batch_script.replace("%%JOBNAME%%", jobname)
                    .replace(
                        "%%TASKS%%",
                        f"srun bash scripts/{RUN_NAME}/task_group/{jobname}.sh &",
                    )
                    # .replace("%%NTASKS%%", str(n_valid_tasks))
                    .replace("%%NTASKS%%", "1")
                    .replace("%%CPUS_PER_TASK%%", "80")
                )
            group_scripts.append(f"scripts/{RUN_NAME}/batch/{jobname}.sh")
            print("-" * 40)
            print("Created task group:", jobname)
            print("-" * 40)

task_status = f"{task_count}/{len(DATASETS)*len(DATA_MODES)*len(DISTILL_METHODS)*len(CLASSIFIERS)}"
group_status = f"{group_count}/{len(DATASETS)*len(DATA_MODES)*len(DISTILL_METHODS)}"
print("-" * 40)
print(
    f"Finished preparing jobs -- {task_status} tasks, {group_status} batch groups",
)
print()
print(f"{task_complete_count}/{task_count} tasks are completed.")
print(f"{group_complete_count}/{group_count} task groups are completed.")
print("-" * 40)


with open(scripts_dir / "run_all.sh", "w") as f:
    f.write("#!/bin/sh\n\n" + "\n".join(f"sbatch {s}" for s in group_scripts))
