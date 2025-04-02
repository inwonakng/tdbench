import time
import warnings

import hydra
import numpy as np
import optuna
import pandas as pd
import pytorch_lightning as pl
import ray
from omegaconf import DictConfig
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import get_scorer
from sklearn.model_selection import StratifiedShuffleSplit, cross_validate
from sklearn.utils.class_weight import compute_sample_weight

from tabdd.config import (
    PipelineConfig,
    load_classifier_train_config,
    load_classifier_tune_configs,
    load_data_mode_config,
    load_dataset_configs,
    load_distill_configs,
    load_encoder_train_config,
    load_encoder_tune_configs,
    load_pipeline_configs,
)
from tabdd.config.paths import RUN_CONFIG_DIR
from tabdd.distill import load_distilled_data
from tabdd.utils import setup_logger

warnings.simplefilter("ignore", category=ConvergenceWarning)

logger = setup_logger()


def get_params(trial: optuna.trial.Trial, tune_params):
    params = {}
    for k, v in tune_params.items():
        val = None
        if v["kind"] == "float":
            val = trial.suggest_float(
                k,
                low=v["lower"],
                high=v["upper"],
            )
        elif v["kind"] == "int":
            val = trial.suggest_int(
                k,
                low=v["lower"],
                high=v["upper"],
            )
        elif v["kind"] == "categorical":
            val = trial.suggest_categorical(k, choices=v["choices"])
        elif v["kind"] == "float_log":
            val = trial.suggest_float(
                k,
                low=v["lower"],
                high=v["upper"],
                log=True,
            )
        else:
            raise NotImplementedError
        params[k] = val
    return params


def tune_classifier(
    pipeline_config: PipelineConfig,
):
    runtime = -1
    per_run_times = []

    data = load_distilled_data(pipeline_config=pipeline_config)
    train_X = data["train"].X
    train_y = data["train"].y
    val_X = data["val"].X
    val_y = data["val"].y

    clf_tune_conf = pipeline_config.classifier_tune_config
    clf_train_conf = pipeline_config.classifier_train_config

    start = time.time()
    # prepare training data for tuning
    if clf_train_conf.n_folds == 0:
        tune_train_X = np.vstack([train_X, val_X])
        tune_train_y = np.hstack([train_y, val_y])
        idxs = np.arange(len(tune_train_X))
        tune_cv = [(idxs[: len(train_X)], idxs[len(train_X) :])]
    else:
        tune_train_X = train_X
        tune_train_y = train_y
        tune_cv = StratifiedShuffleSplit(
            n_splits=clf_train_conf.n_folds,
            random_state=0,
            test_size=1 / clf_train_conf.n_folds,
        )

    sample_weight = compute_sample_weight(
        class_weight="balanced",
        y=tune_train_y,
    )

    scorer = get_scorer(clf_train_conf.metric_name)

    fit_params = dict()
    if clf_tune_conf.use_sample_weight:
        fit_params["sample_weight"] = sample_weight

    def objective(trial: optuna.trial.Trial):
        clf = clf_tune_conf.instantiate(
            {
                **get_params(trial, clf_tune_conf.tune_params),
                "sample_dset": data["train - original"],
            }
        )
        cv_result = cross_validate(
            estimator=clf,
            X=tune_train_X,
            y=tune_train_y,
            cv=tune_cv,
            scoring=scorer,
            fit_params=fit_params,
            n_jobs=1,
        )
        score = cv_result["test_score"].mean()
        return score

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=clf_train_conf.num_samples, n_jobs=1)

    best_params = study.best_params
    results = study.trials_dataframe()

    logger.info(
        f"tuner returned with {len(results)}/{pipeline_config.classifier_train_config.num_samples} runs"
    )
    pipeline_config.save_search_log(results)
    results["duration_sec"]=results["duration"].apply(lambda x: x.total_seconds())
    per_run_times = results["duration_sec"].tolist()
    runtime = sum(per_run_times)
    return best_params, {"ray_run_time": runtime, "per_run_time": per_run_times}


def run_tune(pipeline_config: PipelineConfig) -> tuple[list[dict[str, any]]]:
    logger.info("starting train")
    runtimes = []
    clf_tune_config = pipeline_config.classifier_tune_config
    clf_train_config = pipeline_config.classifier_train_config

    pl.seed_everything(0)
    start = time.time()
    data = load_distilled_data(pipeline_config=pipeline_config)
    end = time.time()
    logger.info("data loaded")

    scorer = get_scorer(clf_train_config.metric_name)
    classifier_params = clf_tune_config.default_params
    sample_weight = compute_sample_weight(
        class_weight="balanced",
        y=data["train"].y,
    )

    if clf_train_config.tune_hyperopt:
        logger.info("starting hparam opt")
        clf = clf_tune_config.instantiate({})
        clf_tune_config.tune_params
        best_params, opt_runtime = tune_classifier(pipeline_config=pipeline_config)
        runtimes.append(
            {"Operation": "Train -- Hyperopt Ray", "Time": opt_runtime["ray_run_time"]}
        )
        runtimes.append(
            {
                "Operation": "Train -- Hyperopt Per Run",
                "Time": opt_runtime["per_run_time"],
            }
        )
        classifier_params.update(best_params)

    if clf_tune_config.use_n_jobs:
        if clf_train_config.tune_hyperopt:
            classifier_params["n_jobs"] = clf_train_config.cpu_per_worker
        else:
            classifier_params["n_jobs"] = -1

    clf = clf_tune_config.instantiate(
        {
            **classifier_params,
            "sample_dset": data["train - original"],
        }
    )

    fit_params = dict(
        X=data["train"].X,
        y=data["train"].y,
    )

    if clf_tune_config.use_sample_weight:
        fit_params["sample_weight"] = sample_weight

    start = time.time()
    clf.fit(**fit_params)
    end = time.time()
    runtimes.append({"Operation": "Train -- Default", "Time": end - start})
    logger.info("classifier trained with default params")

    result = []
    for subset_name, subset in data.items():
        subset_name = " - ".join([w.capitalize() for w in subset_name.split(" - ")])
        start = time.time()
        score = scorer(clf, subset.X, subset.y)
        end = time.time()
        runtimes.append(
            {
                "Operation": f"Inference -- {subset_name}",
                "Time": end - start,
            }
        )
        result.append(
            {
                "Subset": subset_name,
                "Score": score,
            }
        )

    pipeline_config.save_report(result)
    logger.info("Results saved")

    if pipeline_config.classifier_train_config.tune_hyperopt:
        pipeline_config.save_best_params(best_params)
        logger.info("Best parameters saved")

    pipeline_config.save_runtime(runtimes)
    logger.info("Runtime log saved")
    del data
    return result, runtimes


@hydra.main(
    version_base=None,
    config_path=RUN_CONFIG_DIR,
    config_name="tune",
)
def run(
    config: DictConfig,
):

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
                    if pipeline_config.is_complete:
                        continue

                    logger.info("=" * 40)
                    logger.info("STARTING...")
                    logger.info("-" * 40)
                    logger.info(
                        f"Pipeline: {pipeline_config.pretty_name}, N={pipeline_config.distill_size} -- {i_p+1}/{len(pipeline_configs)}"
                    )
                    logger.info(
                        f"Dataset: {dataset_config.dataset_name} -- {i_d+1}/{len(dataset_configs)}"
                    )
                    logger.info(
                        f"Classifier: {classifier_tune_config.classifier_name} [Tune={classifier_train_config.tune_hyperopt}] -- {i_c+1}/{len(classifier_tune_configs)}"
                    )
                    logger.info("-" * 40)
                    logger.info("")

                    result, runtimes = run_tune(pipeline_config=pipeline_config)
                    if result is not None:
                        result = pd.DataFrame(result)
                        test_score = result[result["Subset"] == "Test"]["Score"].mean()
                    if runtimes is not None:
                        runtimes = pd.DataFrame(runtimes)
                        if classifier_train_config.tune_hyperopt:
                            runtimes = runtimes[
                                ~runtimes["Operation"].isin(
                                    ["Train -- Hyperopt Per Run"]
                                )
                            ]
                        time_spent = runtimes["Time"].sum()
                    if result is not None and runtimes is not None:
                        logger.info("-" * 40)
                        logger.info("FINISHED...")
                        logger.info("-" * 40)
                        logger.info(f"Runtime: {time_spent:.2f} seconds")
                        logger.info(f"Test performance: {test_score:.4f}")
                        logger.info("=" * 40)
                    else:
                        logger.info("-" * 40)
                        logger.info("Already complete. Skipping...")
                        logger.info("=" * 40)

    logger.info("###################")
    logger.info("#### ALL DONE! ####")
    logger.info("###################")

    if ray.is_initialized():
        ray.shutdown()
        time.sleep(3)


if __name__ == "__main__":
    run()
