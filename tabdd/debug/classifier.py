import hydra
from omegaconf import DictConfig
from pathlib import Path
import json

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
from tabdd.config.paths import RUN_CONFIG_DIR


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
                    runtime_dir = Path(pipeline_config.runtime_dir)
                    report_dir = Path(pipeline_config.report_dir)
                    params_dir = Path(pipeline_config.params_dir)
                    search_log_dir = None
                    if classifier_train_config.tune_hyperopt:
                        search_log_dir = Path(pipeline_config.search_log_dir)

                    runtime = pipeline_config.load_runtime()
                    distill_time = pipeline_config.load_distill_time()

                    print(
                        "DEBUGGING...\n"
                        + ("-" * 40 + "\n")
                        + f"Pipeline: {pipeline_config.pretty_name}, N={pipeline_config.distill_size} -- {i_p+1}/{len(pipeline_configs)}\n"
                        + f"Dataset: {dataset_config.dataset_name} -- {i_d+1}/{len(dataset_configs)}\n"
                        + f"Classifier: {classifier_tune_config.classifier_name} [Tune={classifier_train_config.tune_hyperopt}] -- {i_c+1}/{len(classifier_tune_configs)}\n"
                        + ("-" * 40 + "\n")
                        + "\n"
                    )

                    runtime_cols = set(runtime.columns)
                    if runtime_cols - set(["Operation", "Time",]):
                        with open(pipeline_config.runtime_dir) as f:
                            json.dump([{**r, **{"Hostname":"npl01"}} for r in runtime])
                    else:
                        if runtime_cols - set(["Operation", "Time", "Hostname"]):
                            print("Something is wrong with runtime!")
                            print(runtime)
                            quit()

                    if isinstance(distill_time, float):
                        print("distill time is float. Converting")
                        with open(pipeline_config.distill_time_dir, "w") as f:
                            json.dump(
                                {"Time": distill_time, "Hostname": "npl01"}, f
                            )
                    else:
                        if not isinstance(distill_time, dict) or set(
                            distill_time.keys()
                        ) - set(["Time", "Hostname"]):
                            print("Something is wrong with distill time!")
                            print(distill_time, type(distill_time))
                            quit()

                    # # if pipeline_config.distill_space == "encoded" and pipeline_config.use_post_data_mode:
                    # if runtime_dir.is_file():
                    #     print(
                    #         "FIXING Runtime...\n"
                    #         + ("-" * 40 + "\n")
                    #         + f"Pipeline: {pipeline_config.pretty_name}, N={pipeline_config.distill_size} -- {i_p+1}/{len(pipeline_configs)}\n"
                    #         + f"Dataset: {dataset_config.dataset_name} -- {i_d+1}/{len(dataset_configs)}\n"
                    #         + f"Classifier: {classifier_tune_config.classifier_name} [Tune={classifier_train_config.tune_hyperopt}] -- {i_c+1}/{len(classifier_tune_configs)}\n"
                    #         + ("-" * 40 + "\n")
                    #         + "\n"
                    #     )
                    #     with open(runtime_dir) as f:
                    #         runtimes = json.load(f)
                    #         try:
                    #             fixed = [
                    #                 {
                    #                     "Operation": r["Operation"],
                    #                     "Time": r["Time"],
                    #                     "Hostname": "npl01",
                    #                 }
                    #                 for r in runtimes
                    #             ]
                    #         except:
                    #             print("somethins is very messed up")
                    #             print(runtime_dir)
                    #             quit()
                    #         with open(runtime_dir, "w") as f:
                    #             json.dump(fixed, f)
                    # if pipeline_config.distill_time_dir is not None:
                    #     print(
                    #         "FIXING distilltime...\n"
                    #         + ("-" * 40 + "\n")
                    #         + f"Pipeline: {pipeline_config.pretty_name}, N={pipeline_config.distill_size} -- {i_p+1}/{len(pipeline_configs)}\n"
                    #         + f"Dataset: {dataset_config.dataset_name} -- {i_d+1}/{len(dataset_configs)}\n"
                    #         + f"Classifier: {classifier_tune_config.classifier_name} [Tune={classifier_train_config.tune_hyperopt}] -- {i_c+1}/{len(classifier_tune_configs)}\n"
                    #         + ("-" * 40 + "\n")
                    #         + "\n"
                    #     )
                    #     if Path(pipeline_config.distill_time_dir).is_file():
                    #         with open(pipeline_config.distill_time_dir) as f:
                    #
                    #             old = json.load(f)
                    #             if not isinstance(old, dict):
                    #                 try:
                    #                     old = float(old)
                    #                 except:
                    #                     print("I cant parse this file")
                    #                     print(pipeline_config.distill_time_dir)
                    #                     quit()
                    #             oldkeys = list(old.keys())
                    #             if not "time" in oldkeys and not "Time" in oldkeys:
                    #                 print("I cant parse this file")
                    #                 print(pipeline_config.distill_time_dir)
                    #                 quit()
                    #             else:
                    #                 if "Time" in oldkeys:
                    #                     old = old["Time"]
                    #                 elif "time" in oldkeys:
                    #                     old = old["time"]
                    #                     if isinstance(old, dict):
                    #                         old = old["time"]
                    #             if not isinstance(old, float):
                    #                 print("somethings is very messed up")
                    #                 print(pipeline_config.distill_time_dir)
                    #                 quit()
                    #             new = {"Hostname": "npl01", "Time": old}
                    #         with open(pipeline_config.distill_time_dir, "w") as f:
                    #             json.dump(new, f)
                    #
                    # else:
                    #     continue


if __name__ == "__main__":
    run()
