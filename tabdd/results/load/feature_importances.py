import pandas as pd

from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

from tabdd.data import load_data_module
from tabdd.config import get_pipeline_configs, DataConfig
from tabdd.tune.classifier import TuneClassifierRun
from tabdd.config.paths import DATA_DIR


def feature_importance(model: XGBClassifier | LogisticRegression):
    if isinstance(model, XGBClassifier):
        return model.feature_importances_
    elif isinstance(model, LogisticRegression):
        return model.coef_[0]
    else:
        raise NotImplementedError(
            f"Feature importance function not defined for {model.__class__.__name__}"
        )


def load_feature_importances(
    dataset_name: str,
    classifier_name: str,
    tune_hyperopt: bool = False,
) -> tuple[pd.DataFrame | None, bool]:
    data_config = DataConfig(
        dataset_name=dataset_name,
        scale_mode="standard",
        parse_mode="onehot",
        n_bins=10,
        bin_strat="uniform",
        batch_size=512,
    )
    dm = load_data_module(data_config)
    dm.prepare_data()

    feature_cols = []
    offset = 0
    for i, f_idx in enumerate(dm.feature_mask):
        if offset > 0 and f_idx != feature_cols[-1][1]:
            offset = 0
        feature_cols += [(i, f_idx, offset)]
        offset += 1

    configs = get_pipeline_configs(dataset_name, classifier_name, "all")
    importance_scores = []
    all_done = True
    for config in configs:
        config.tune_hyperopt = tune_hyperopt
        run = TuneClassifierRun(config)
        if not run.is_complete:
            all_done = False
            continue

        importance_scores += [
            {
                "Dataset": config.distill_config.data_config.dataset_name,
                "Data Mode": config.distill_config.pretty_name,
                "Distill Size": config.distill_config.distill_size,
                "Distill Method": config.distill_config.distill_method,
                "Distill Space": config.distill_config.distill_space,
                "Encoder": config.distill_config.encoder_name,
                "Output Space": config.distill_config.output_space,
                "Convert Binary": config.distill_config.convert_binary,
                "Cluster Center": config.distill_config.cluster_center,
                "Binary Index": i,
                "Original Feature": f_idx,
                "Binary Feature": offset,
                "Importance Score": f_imp,
            }
            for (i, f_idx, offset), f_imp in zip(
                feature_cols, feature_importance(run.load_model())
            )
        ]

    if len(importance_scores) == 0:
        return None, False

    importance_scores = pd.DataFrame(importance_scores)

    return importance_scores, all_done
