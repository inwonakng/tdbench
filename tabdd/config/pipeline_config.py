import itertools
import pandas as pd
from typing import Literal
from pathlib import Path
import json
from dataclasses import dataclass

from .classifier_config import ClassifierTuneConfig, ClassifierTrainConfig
from .encoder_config import (
    EncoderTuneConfig,
    MultiEncoderTuneConfig,
    EncoderTrainConfig,
)
from .distill_config import DistillConfig
from .dataset_config import DatasetConfig
from .data_mode_config import DataModeConfig
from tabdd.config.hostname import HOSTNAME


@dataclass
class PipelineConfig:
    dataset_config: DatasetConfig
    data_mode_config: DataModeConfig
    classifier_tune_config: ClassifierTuneConfig
    classifier_train_config: ClassifierTrainConfig
    distill_config: DistillConfig
    encoder_tune_config: EncoderTuneConfig | MultiEncoderTuneConfig = None
    encoder_train_config: EncoderTrainConfig = None
    cluster_center: Literal["closest", "centroid"] = "centroid"
    distill_size: int = 0
    distill_space: Literal["original", "encoded"] = None
    output_space: Literal["original", "encoded", "decoded"] = "original"
    convert_binary: bool = False
    random_seed: int = 0
    rerun_tune: bool = False
    save_dir: Path = None
    params_dir: str = None
    report_dir: str = None
    distill_time_dir: str = None
    runtime_dir: str = None
    search_log_dir: str = None

    def set(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

    def __post_init__(self):
        self.save_dir = self.classifier_train_config.results_dir / self.run_name
        self.save_dir.mkdir(parents=True, exist_ok=True)

        prefix = "default"
        if self.classifier_train_config.tune_hyperopt:
            prefix = "tuned"
        self.report_dir = str(
            self.save_dir / f"{prefix}_report_{self.random_seed:03}.json"
        )
        self.params_dir = str(
            self.save_dir / f"{prefix}_params_{self.random_seed:03}.json"
        )
        self.runtime_dir = str(
            self.save_dir / f"{prefix}_runtime_{self.random_seed:03}.json"
        )
        self.distill_time_dir = (
            self.save_dir / f"{prefix}_distill_{self.random_seed:03}.json"
        )
        if self.classifier_train_config.tune_hyperopt:
            self.search_log_dir = str(
                self.save_dir / f"{prefix}_search_log_{self.random_seed:03}.json"
            )

    @property
    def distill_method_name(self) -> str:
        return self.distill_config.distill_method_name

    @property
    def distill_method_pretty_name(self) -> str:
        return self.distill_config.pretty_name

    @property
    def dataset_name(self) -> str:
        return self.dataset_config.dataset_name

    @property
    def classifier_name(self) -> str:
        return self.classifier_tune_config.classifier_name

    @property
    def attributes(self):
        attr_mapping = {
            "Dataset": self.dataset_config.dataset_name,
            "Classifier": self.classifier_tune_config.classifier_name,
            "N": self.distill_size,
            "Data Parse Mode": self.data_mode_config.parse_mode,
            "Post Data Parse Mode": (
                self.distill_config.post_data_mode_name
                if self.distill_config.post_data_mode_name is not None
                else self.data_mode_config.parse_mode
            ),
            "Data Mode": self.pretty_name,
            "Distill Method": self.distill_config.pretty_name,
            "Encoder": self.encoder_pretty_name,
            "Distill Space": (
                self.distill_space if self.distill_space is not None else "N/A"
            ),
            "Output Space": self.output_space,
            "Convert Binary": self.convert_binary,
            "Cluster Center": self.cluster_center,
        }
        return attr_mapping

    @property
    def encoder_pretty_name(self) -> str:
        if self.encoder_tune_config is None:
            return "N/A"
        else:
            return self.encoder_tune_config.pretty_name

    @property
    def run_name(self):
        run_name = "{}/{}/{}/{}/{}/{}_folds/{}".format(
            self.classifier_tune_config.classifier_name.lower(),
            self.dataset_config.identifier,
            self.data_mode_config.identifier,
            self.classifier_train_config.optimizer_name,
            self.classifier_train_config.metric_name,
            self.classifier_train_config.n_folds,
            self.identifier,
        )
        return run_name

    @property
    def distilled_data_dir(self):
        return (
            self.classifier_train_config.results_dir
            / "distilled_data"
            / self.dataset_config.identifier
            / self.data_mode_config.identifier
            / self.identifier
            / f"distilled_{self.random_seed:03}.h5"
        )

    @property
    def identifier(self):
        data_mode = ""
        if self.distill_method_name in [
            "original",
            "encoded",
            "decoded",
            "random_sample",
        ]:
            data_mode += self.distill_method_name
        else:
            if self.distill_space == "original":
                data_mode = f"{self.distill_method_name}_{self.distill_space}"
            elif self.distill_space == "encoded":
                data_mode = f"{self.distill_method_name}_{self.distill_space}_{self.output_space}"
                if self.output_space == "decoded" and self.convert_binary:
                    data_mode += "_binary"
            else:
                raise NotImplementedError(
                    f"Distill method [{self.distill_method_name}], Distill space[{self.distill_space}] not found"
                )
            data_mode += f"/{self.cluster_center}"

        if self.distill_config.post_data_mode_config is not None:
            data_mode += (
                f"/post_process/{self.distill_config.post_data_mode_config.identifier}"
            )

        if "encoded" in data_mode or "decoded" in data_mode:
            data_mode += f"/{self.encoder_tune_config.identifier}/{str(self.encoder_train_config.latent_dim)}"

        if self.distill_config.can_distill:
            data_mode += f"/N={self.distill_size}"

        return data_mode

    @property
    def pretty_name(self):
        pretty_name = f"{self.data_mode_config.parse_mode.capitalize()} -> "
        # if self.data_mode_config.parse_mode == "mixed":
        # pretty_name += "Mixed "
        if self.distill_method_name in ["original", "random_sample"]:
            pretty_name += self.distill_config.pretty_name
        elif self.distill_method_name in ["encoded", "decoded"]:
            pretty_name += f"{self.encoder_tune_config.pretty_name} {self.distill_config.pretty_name}"
        else:
            pretty_name += "[ "
            if self.distill_space == "encoded":
                pretty_name += f"{self.encoder_tune_config.pretty_name}-Encoded => "
            pretty_name += self.distill_config.pretty_name
            if self.distill_config.is_cluster:
                pretty_name += f" / {self.cluster_center.capitalize()}"
            pretty_name += f" => {self.output_space.capitalize()}"
            if self.convert_binary and self.output_space == "decoded":
                pretty_name += "-Binary"
            pretty_name += " ]"

        if self.distill_config.post_data_mode_config is not None:
            pretty_name += f" -> {self.distill_config.post_data_mode_config.parse_mode.capitalize()}"
        else:
            pretty_name += f" -> {self.data_mode_config.parse_mode.capitalize()}"

        return pretty_name

    @property
    def use_post_data_mode(self):
        return (
            self.distill_config.post_data_mode_config is not None
            # not applicable if the output space is in latent
            and not self.output_space == "encoded"
            and (
                # only apply when the post data mode is different from the original data mode
                self.data_mode_config.identifier
                != self.distill_config.post_data_mode_config.identifier
            )
        )

    def save_distill_time(self, time: float):
        self.distill_time_dir.parent.mkdir(parents=True, exist_ok=True)
        with open(self.distill_time_dir, "w") as f:
            json.dump({"Hostname": HOSTNAME, "Time": time}, f)

    def load_distill_time(self):
        distill_time = {"Hostname": "Unknown", "Time": -1}
        if Path(self.distill_time_dir).is_file():
            try:
                with open(self.distill_time_dir) as f:
                    distill_time = json.load(f)
                # if its not formated right, fix
                if not isinstance(distill_time, dict) and isinstance(
                    distill_time, (int, float)
                ):
                    distill_time = {"Hostname": "npl", "Time": distill_time}

                with open(self.distill_time_dir, "w") as f:
                    json.dump(distill_time, f)
            except:
                pass
        return distill_time

    @property
    def is_complete(self):
        if self.classifier_train_config.tune_hyperopt:
            return (
                Path(self.report_dir).is_file()
                and Path(self.params_dir).is_file()
                and Path(self.search_log_dir).is_file()
            )
        else:
            return Path(self.report_dir).is_file()

    def save_best_params(self, best_params: dict[any]):
        if not self.classifier_train_config.tune_hyperopt:
            return
        Path(self.params_dir).parent.mkdir(parents=True, exist_ok=True)
        with open(self.params_dir, "w") as f:
            json.dump(best_params, f, indent=2)

    def load_best_params(self):
        if not self.tune_hyperopt:
            return None
        with open(self.params_dir) as f:
            best_params = json.load(f)
        return best_params

    def save_report(self, report: list[dict[str, float]]):
        Path(self.report_dir).parent.mkdir(parents=True, exist_ok=True)
        with open(self.report_dir, "w") as f:
            json.dump(report, f, indent=2)

    def load_report(self) -> pd.DataFrame:
        try:
            with open(self.report_dir) as f:
                report = pd.DataFrame(json.load(f))
                return report
        except:
            raise Exception(f"Unable to open report at {self.report_dir}")

    def save_runtime(self, runtimes: list[dict[str, float]]):
        Path(self.runtime_dir).parent.mkdir(parents=True, exist_ok=True)
        with open(self.runtime_dir, "w") as f:
            json.dump([{**r, **{"Hostname": HOSTNAME}} for r in runtimes], f, indent=2)

    def load_runtime(self) -> pd.DataFrame:
        try:
            with open(self.runtime_dir) as f:
                # runtime = json.load(f)
                runtime = pd.DataFrame(json.load(f))
                # runtime[runtime.index[-1]]

                if "Train -- Hyperopt Per Run" in runtime["Operation"].values:

                    new_row = pd.DataFrame([{
                        "Operation": "Train -- Hyperopt Total",
                        "Time": sum(
                            runtime[
                                runtime["Operation"] == "Train -- Hyperopt Per Run"
                            ]["Time"].values[0]
                        ),
                        "Hostname": HOSTNAME,
                    }])
                    runtime = pd.concat([runtime, new_row])
                return runtime
        except Exception as e:
            print(self.runtime_dir)
            raise e
            # return pd.DataFrame(
            #     [
            #         {"Operation": "DataDistill", "Time": -1, "Hostname": "Unknown"},
            #         {
            #             "Operation": "Train -- Hyperopt Ray",
            #             "Time": -1,
            #             "Hostname": "Unknown",
            #         },
            #         {
            #             "Operation": "Train -- Hyperopt Per Run",
            #             "Time": [-1],
            #             "Hostname": "Unknown",
            #         },
            #         {
            #             "Operation": "Train -- Hyperopt Total",
            #             "Time": -1,
            #             "Hostname": "Unknown",
            #         },
            #         {
            #             "Operation": "Train -- Default",
            #             "Time": -1,
            #             "Hostname": "Unknown",
            #         },
            #         {
            #             "Operation": "Inference -- Train",
            #             "Time": -1,
            #             "Hostname": "Unknown",
            #         },
            #         {
            #             "Operation": "Inference -- Train - Original",
            #             "Time": -1,
            #             "Hostname": "Unknown",
            #         },
            #         {
            #             "Operation": "Inference -- Val",
            #             "Time": -1,
            #             "Hostname": "Unknown",
            #         },
            #         {
            #             "Operation": "Inference -- Test",
            #             "Time": -1,
            #             "Hostname": "Unknown",
            #         },
            #         {
            #             "Operation": "Train -- Hyperopt Total",
            #             "Time": -1,
            #             "Hostname": "Unknown",
            #         },
            #     ]
            # )

    def save_search_log(self, log: pd.DataFrame) -> None:
        Path(self.search_log_dir).parent.mkdir(parents=True, exist_ok=True)
        log.to_json(self.search_log_dir, orient="records", index=False)

    def load_search_log(self) -> None | pd.DataFrame:
        if not self.classifier_train_config.tune_hyperopt:
            return None
        return pd.read_json(self.search_log_dir)

    def load_report_w_runtime(self) -> pd.DataFrame:
        report = self.load_report()
        runtimes = self.load_runtime()

        tr_time_def = runtimes[runtimes["Operation"] == "Train -- Default"][
            "Time"
        ].values[0]
        tr_time_opt = 0
        tr_time_opt_ray = 0
        if self.classifier_train_config.tune_hyperopt:
            tr_time_opt = runtimes[runtimes["Operation"] == "Train -- Hyperopt Total"][
                "Time"
            ].values[0]
            tr_time_opt_ray = runtimes[
                runtimes["Operation"] == "Train -- Hyperopt Ray"
            ]["Time"].values[0]

        report["Default Train Time"] = tr_time_def
        report["Opt Train Time"] = tr_time_opt_ray
        report["Opt Train Time Total"] = tr_time_opt
        if "Hostname" in runtimes.columns:
            report["Hostname"] = runtimes["Hostname"].values[0]
        else:
            report["Hostname"] = "Unknown"
        report["Data Distill Time"] = self.load_distill_time()["Time"]

        report["Inference Time"] = [
            runtimes[runtimes["Operation"] == f"Inference -- {subset}"]["Time"].values[
                0
            ]
            for subset in report["Subset"].values
        ]

        return report


def load_pipeline_configs(
    dataset_config: DatasetConfig,
    data_mode_config: DataModeConfig,
    classifier_tune_config: ClassifierTuneConfig,
    classifier_train_config: ClassifierTrainConfig,
    distill_config: DistillConfig,
    encoder_train_config: EncoderTrainConfig,
    encoder_tune_configs: list[EncoderTuneConfig],
) -> list[PipelineConfig]:
    variations = dict()

    if distill_config.can_distill:
        variations["distill_size"] = distill_config.distill_sizes

    # Logic for valid distill configs
    if distill_config.can_use_encoder:
        # if distill_config.is_baseline:
        #     variations["distill_space"] = ["encoded"]
        # else:
        #     variations["distill_space"] = ["original", "encoded"]
        if len(encoder_tune_configs) > 0:
            variations["distill_space"] = ["encoded"]
            if "original" in distill_config.distill_spaces:
                variations["distill_space"] += ["original"]
        else:
            variations["distill_space"] = ["original"]

    if distill_config.is_random:
        variations["random_seed"] = list(range(distill_config.random_iters))

    if distill_config.is_cluster:
        variations["cluster_center"] = ["centroid", "closest"]

    all_configs = [
        dict(
            dataset_config=dataset_config,
            data_mode_config=data_mode_config,
            distill_config=distill_config,
            classifier_tune_config=classifier_tune_config,
            classifier_train_config=classifier_train_config,
            **dict(zip(variations.keys(), v)),
        )
        for v in itertools.product(*variations.values())
    ]

    # set output space if encoded
    if distill_config.can_use_encoder:
        with_encoder = []
        for conf in all_configs:
            if conf["distill_space"] == "encoded":
                with_encoder += [
                    dict(
                        **conf,
                        encoder_tune_config=econf,
                        encoder_train_config=encoder_train_config,
                    )
                    for econf in encoder_tune_configs
                ]
            else:
                with_encoder += [conf]

        with_output_space = []
        for conf in with_encoder:
            if conf["distill_space"] == "encoded" and not distill_config.is_baseline:
                if "decoded" in distill_config.output_spaces:
                    with_output_space += [
                        dict(
                            **conf,
                            output_space="decoded",
                        ),
                        dict(
                            **conf,
                            output_space="decoded",
                            convert_binary=True,
                        ),
                    ]
                if (
                    distill_config.post_data_mode_config is None
                    and "encoded" in distill_config.output_spaces
                ):
                    with_output_space.append(
                        dict(
                            **conf,
                            output_space="encoded",
                        )
                    )

                # clustering methods can also handle using the original space (by clustering idx)
                if ( 
                    distill_config.is_cluster
                    and "original" in distill_config.output_spaces
                ) and not (
                    conf["distill_space"] == "encoded"
                    and conf["cluster_center"] != "closest"
                ):
                    with_output_space.append(
                        dict(
                            **conf,
                            output_space="original",
                        )
                    )
            else:
                with_output_space += [conf]
        all_configs = with_output_space
    all_pipeline_configs = [PipelineConfig(**conf) for conf in all_configs]

    return all_pipeline_configs
