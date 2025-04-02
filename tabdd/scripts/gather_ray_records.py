import hydra
from omegaconf import DictConfig
import time
import time
from pathlib import Path
import pandas as pd
import string
import random


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
from tabdd.config.paths import RAY_TMP_DIR, RUN_CONFIG_DIR


@hydra.main(
    version_base=None, config_path=RUN_CONFIG_DIR, config_name="tune_classifier"
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
                    search_log_dir = Path(pipeline_config.search_log_dir)


                    if (runtime_dir.is_file()
                        and report_dir.is_file() 
                        and not search_log_dir.is_file()
                    ):

                        ray_dir = (
                            Path(pipeline_config.classifier_train_config.storage_dir)
                            / f"{pipeline_config.run_name}
                            / random_seed={pipeline_config.random_seed:03}"
                        )
                        df = pd.concat([
                            pd.read_csv(l) for l in ray_dir.glob("*/*.csv")
                        ])
                        df = df.sort_values("timestamp")
                        df.to_json(search_log_dir, orient="records", index=False)

                    continue

if __name__ == "__main__":
    run()
