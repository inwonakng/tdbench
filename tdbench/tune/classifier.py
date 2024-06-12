import hydra
from omegaconf import DictConfig
import pytorch_lightning as pl
import os
import numpy as np
import json
import time
from sklearn.metrics import get_scorer
from sklearn.model_selection import StratifiedShuffleSplit, cross_validate
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.exceptions import ConvergenceWarning
import warnings
import time
from pathlib import Path
import pandas as pd
import string
import random

import ray
from ray import tune, air
from ray.tune.search.hyperopt import HyperOptSearch

from tdbench.distill import load_distilled_data

from tdbench.utils import setup_logger
from tdbench.config import (
    load_dataset_configs,
    load_data_mode_config,
    load_distill_configs,
    load_classifier_tune_configs,
    load_classifier_train_config,
    load_encoder_tune_configs,
    load_encoder_train_config,
    load_pipeline_configs,
    PipelineConfig,
)
from tdbench.config.paths import RAY_TMP_DIR, RUN_CONFIG_DIR

warnings.simplefilter("ignore", category=ConvergenceWarning)

logger = setup_logger()

def train_classifier(
    config: dict[str, any],
    pipeline_config: PipelineConfig,
):
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
    clf = clf_tune_conf.instantiate({
        **config,
        "sample_dset": data["train - original"],
    })

    fit_params = dict()
    if clf_tune_conf.use_sample_weight:
        fit_params["sample_weight"] = sample_weight

    score = cross_validate(
        estimator=clf,
        X=tune_train_X,
        y=tune_train_y,
        cv=tune_cv,
        scoring=scorer,
        fit_params=fit_params,
    )
    # if time.time() - start < 3:
    #     time.sleep(1)
    air.session.report({"score": score["test_score"].mean()})


def _tune_classifier(
    pipeline_config: PipelineConfig,
    run_name: str,
):

    # define train function
    if pipeline_config.classifier_train_config.optimizer_name == "hyperopt":
        search_alg = HyperOptSearch(metric="score", mode="max", random_state_seed=0)
    else:
        raise NotImplementedError

    best_config = None
    runtime = -1
    per_run_times = []
    tuner = tune.Tuner(
        tune.with_resources(
            tune.with_parameters(
                train_classifier,
                pipeline_config=pipeline_config,
            ),
            resources={"CPU": pipeline_config.classifier_train_config.cpu_per_worker},
        ),
        tune_config=tune.TuneConfig(
            metric="score",
            mode="max",
            num_samples=pipeline_config.classifier_train_config.num_samples,
            search_alg=search_alg,
            reuse_actors=True,
            max_concurrent_trials=pipeline_config.classifier_train_config.max_concurrent_trials,
        ),
        run_config=air.RunConfig(
            verbose=0,
            storage_path=pipeline_config.classifier_train_config.storage_dir,
            name=run_name,
            failure_config=air.FailureConfig(
                fail_fast=True,
                # max_failures=0,
            ),
        ),
        param_space=pipeline_config.classifier_tune_config.tune_params,
    )
    try:
        start = time.time()
        results = tuner.fit()
        end = time.time()
        logger.info(
            f"tuner returned with {len(results)}/{pipeline_config.classifier_train_config.num_samples} runs"
        )
        skipped_runs = pipeline_config.classifier_train_config.num_samples - len(
            results
        )
        num_errors = sum(r.error is not None for r in results)
        if skipped_runs > 0:
            logger.info(f"tuner skipped {skipped_runs} runs.. didn't finish")
            raise Exception("Ray tune is not complete.")
        elif num_errors > 0:
            logger.info(f"tuner has {num_errors} errors.. didn't finish")
            raise Exception("Ray tune is not complete.")
        else:
            best_result = results.get_best_result(metric="score", mode="max")
            best_config = best_result.config
            runtime = end - start
            search_log = pd.concat(
                [pd.read_csv(r.log_dir / "progress.csv") for r in results]
            )
            search_log = search_log.sort_values("timestamp")
            per_run_times = search_log["time_this_iter_s"].tolist()
            pipeline_config.save_search_log(search_log)

    except:
        logger.info("Ray run failed")

    return best_config, {"ray_run_time": runtime, "per_run_time": per_run_times}


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
        best_params = None
        while best_params is None:
            # try till ray actually works.
            best_params, opt_runtime = _tune_classifier(
                pipeline_config=pipeline_config,
                run_name=f"{pipeline_config.run_name}/random_seed={pipeline_config.random_seed:03}",
            )
            if best_params is None:
                time.sleep(3)
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

    clf = clf_tune_config.instantiate({
        **classifier_params,
        "sample_dset": data["train - original"],
    })

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

    ray_tmpdir = (
        RAY_TMP_DIR
        + "/"
        + "".join(random.choices(string.ascii_uppercase + string.digits, k=6))
    )
    if classifier_train_config.tune_hyperopt:
        os.environ["RAY_TMPDIR"] = ray_tmpdir
        ray.init(
            address="local",
            _temp_dir=ray_tmpdir,
            _enable_object_reconstruction=False,
        )
        time.sleep(5)

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

                    # if (
                    #     not pipeline_config.classifier_train_config.rerun_tune
                    #     and pipeline_config.is_complete
                    # ):
                    #     if not Path(pipeline_config.distill_time_dir).is_file():
                    #         data = load_distilled_data(pipeline_config=pipeline_config)
                    #     continue

                    logger.info("=" * 40)
                    logger.info("STARTING...")
                    logger.info("-" * 40)
                    logger.info(f"Pipeline: {pipeline_config.pretty_name}, N={pipeline_config.distill_size} -- {i_p+1}/{len(pipeline_configs)}")
                    logger.info(f"Dataset: {dataset_config.dataset_name} -- {i_d+1}/{len(dataset_configs)}")
                    logger.info(f"Classifier: {classifier_tune_config.classifier_name} [Tune={classifier_train_config.tune_hyperopt}] -- {i_c+1}/{len(classifier_tune_configs)}")
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
