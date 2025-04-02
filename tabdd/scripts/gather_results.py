import pandas as pd
import hydra
import numpy as np

from tabdd.utils import progress_bar
from tabdd.results.load import load_all_clf_perf

# REPORTS_DIR = "hpo_measure"
# REPORTS_DIR = "datm"
REPORTS_DIR = "tabpfn"
TUNE_HYPEROPT = "false"


def compute_regret(results):
    results["Scaled Regret with RS"] = -np.inf
    results["Regret"] = -np.inf
    results["Scaled Regret"] = -np.inf

    with progress_bar() as progress:
        grouped_res = list(
            results[results["Subset"] == "Test"].groupby(["Dataset", "Classifier", "N"])
        )
        task = progress.add_task("Computing Regret", total=len(grouped_res))

        ori_scores = {}
        for (ds, clf), grouped in results[
            (results["Subset"] == "Test")
            & (results["Data Parse Mode"] == "mixed")
            & (results["Distill Method"] == "Original")
        ].groupby(
            [
                "Dataset",
                "Classifier",
            ]
        ):
            if ds not in ori_scores:
                ori_scores[ds] = {}
            if clf not in ori_scores[ds]:
                ori_scores[ds][clf] = None
            if len(grouped) > 0:
                ori_scores[ds][clf] = grouped["Score"].mean()

        rs_scores = {}
        for (ds, clf, n), grouped in results[
            (results["Subset"] == "Test")
            & (results["Data Parse Mode"] == "mixed")
            & (results["Distill Method"] == "Random Sample")
        ].groupby(
            [
                "Dataset",
                "Classifier",
                "N",
            ]
        ):
            if ds not in rs_scores:
                rs_scores[ds] = {}
            if clf not in rs_scores[ds]:
                rs_scores[ds][clf] = {}
            if n not in rs_scores[ds][clf]:
                rs_scores[ds][clf][n] = None
            if len(grouped) > 0:
                rs_scores[ds][clf][n] = grouped["Score"].mean()

        for (ds, clf, n), grouped in grouped_res:
            scores = results[
                (results["Dataset"] == ds)
                & (results["Classifier"] == clf)
                & (results["N"] == n)
                & (results["Subset"] == "Test")
            ]["Score"]
            if len(scores) == 0:
                progress.update(task, advance=1)
                continue

            if ds not in ori_scores or clf not in ori_scores[ds]:
                progress.update(task, advance=1)
                continue
            ori_score = ori_scores[ds][clf]
            if ori_score is None:
                continue

            results.loc[
                (
                    (results["Dataset"] == ds)
                    & (results["Classifier"] == clf)
                    & (results["N"] == n)
                    & (results["Subset"] == "Test")
                ),
                "Regret",
            ] = (
                ori_score - scores
            )

            results.loc[
                (
                    (results["Dataset"] == ds)
                    & (results["Classifier"] == clf)
                    & (results["N"] == n)
                    & (results["Subset"] == "Test")
                ),
                "Scaled Regret",
            ] = (ori_score - scores) / ori_score

            if (
                ds not in rs_scores
                or clf not in rs_scores[ds]
                or n not in rs_scores[ds][clf]
            ):
                progress.update(task, advance=1)
                continue
            rs_score = rs_scores[ds][clf][n]
            if rs_score is None:
                continue

            results.loc[
                (
                    (results["Dataset"] == ds)
                    & (results["Classifier"] == clf)
                    & (results["N"] == n)
                    & (results["Subset"] == "Test")
                ),
                "Scaled Regret with RS",
            ] = (ori_score - scores) / (rs_score - scores + 1e-8)

            progress.update(task, advance=1)
    return results


# we need to use the compose API because we are considering multiple configs
def load_df():

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

    classifiers = [
        # "xgb",
        # "ft_transformer",
        # "resnet",
        # "mlp",
        # "logistic_regression",
        # "gaussian_nb",
        # "knn",
        "tabpfn",
    ]

    to_concat = []

    """
    print("gathering mixed...")
    mixed_config = hydra.compose(
        config_name="tune",
        overrides=[
            f"data/datasets=[{','.join(datasets)}]",
            "data.mode.parse_mode=mixed",
            # "distill/methods=[original,random_sample,kmeans,agglo,kip,gm]",
            "distill/methods=[original]",
            f"classifier/models=[{','.join(classifiers)}]",
            # f"classifier/models=[xgb]",
            "encoder/models=[]",
            f"classifier.train.tune_hyperopt={TUNE_HYPEROPT}",
            f"classifier.train.results_dir={REPORTS_DIR}",
        ],
    )
    mixed_report, mixed_incomplete = load_all_clf_perf(mixed_config, refresh=True)
    mixed_report.to_csv("mixed_report.csv", index=False)
    mixed_incomplete.to_csv("mixed_incomplete.csv", index=False)
    mixed_rep_ids = mixed_report[["Dataset", "Classifier", "Data Mode"]].apply(
        lambda x: " ".join(x), axis=1
    )
    mixed_incomp_ids = mixed_incomplete[["Dataset", "Classifier", "Data Mode"]].apply(
        lambda x: " ".join(x), axis=1
    )
    to_concat.append(mixed_report[~mixed_rep_ids.isin(mixed_incomp_ids)])
    print("got mixed...")
    """

    print("gathering onehot...")
    onehot_config = hydra.compose(
        config_name="tune",
        overrides=[
            f"data/datasets=[{','.join(datasets)}]",
            "data.mode.parse_mode=onehot",
            # "distill/methods=[original,random_sample,kmeans,agglo,kip,gm]",
            "distill/methods=[kmeans]",
            # "distill/common=n_100",
            f"classifier/models=[{','.join(classifiers)}]",
            # "encoder/models=[mlp,gnn,tf]",
            "encoder/models=[tf]",
            ### only include this if we don't want vanilla version at all.
            "encoder.train.train_target=[multihead]", 
            f"classifier.train.tune_hyperopt={TUNE_HYPEROPT}",
            f"classifier.train.results_dir={REPORTS_DIR}",
        ],
    )
    onehot_report, onehot_incomplete = load_all_clf_perf(onehot_config, refresh=True)
    onehot_report.to_csv("onehot_report.csv", index=False)
    onehot_incomplete.to_csv("onehot_incomplete.csv", index=False)
    onehot_rep_ids = onehot_report[["Dataset", "Classifier", "Data Mode"]].apply(
        lambda x: " ".join(x), axis=1
    )
    onehot_incomp_ids = onehot_incomplete[["Dataset", "Classifier", "Data Mode"]].apply(
        lambda x: " ".join(x), axis=1
    )
    to_concat.append(onehot_report[~onehot_rep_ids.isin(onehot_incomp_ids)])
    print("got onehot...")

    """
    print("gathering mixed -> onehot...")
    mixed_onehot_config = hydra.compose(
        config_name="tune",
        overrides=[
            f"data/datasets=[{','.join(datasets)}]",
            "data.mode.parse_mode=mixed",
            "distill/methods=[random_sample,kmeans,agglo,kip,gm]",
            "+distill.common.post_data_mode_name=onehot",
            f"classifier/models=[{','.join(classifiers)}]",
            "encoder/models=[]",
            f"classifier.train.tune_hyperopt={TUNE_HYPEROPT}",
            f"classifier.train.results_dir={REPORTS_DIR}",
        ],
    )
    mixed_onehot_report, mixed_onehot_incomplete = load_all_clf_perf(
        mixed_onehot_config, refresh=True
    )
    mixed_onehot_report.to_csv("mixed_onehot_report.csv", index=False)
    mixed_onehot_incomplete.to_csv("mixed_onehot_incomplete.csv", index=False)
    mixed_onehot_rep_ids = mixed_onehot_report[
        ["Dataset", "Classifier", "Data Mode"]
    ].apply(lambda x: " ".join(x), axis=1)
    mixed_onehot_incomp_ids = mixed_onehot_incomplete[
        ["Dataset", "Classifier", "Data Mode"]
    ].apply(lambda x: " ".join(x), axis=1)
    to_concat.append(
        mixed_onehot_report[~mixed_onehot_rep_ids.isin(mixed_onehot_incomp_ids)]
    )
    print("got mixed -> onehot...")
    """

    """
    print("gathering onehot -> mixed...")
    onehot_mixed_config = hydra.compose(
        config_name="tune",
        overrides=[
            f"data/datasets=[{','.join(datasets)}]",
            "data.mode.parse_mode=onehot",
            # "distill/methods=[random_sample,kmeans,agglo,kip,gm]",
            "distill/methods=[kmeans]",
            "distill/common=n_100",
            "+distill.common.post_data_mode_name=mixed",
            f"classifier/models=[{','.join(classifiers)}]",
            # "encoder/models=[mlp,gnn,tf]",
            "encoder/models=[tf]",
            f"classifier.train.tune_hyperopt={TUNE_HYPEROPT}",
            f"classifier.train.results_dir={REPORTS_DIR}",
        ],
    )
    onehot_mixed_report, onehot_mixed_incomplete = load_all_clf_perf(
        onehot_mixed_config, refresh=True
    )
    onehot_mixed_report.to_csv("onehot_mixed_report.csv", index=False)
    onehot_mixed_incomplete.to_csv("onehot_mixed_incomplete.csv", index=False)
    onehot_mixed_rep_ids = onehot_mixed_report[
        ["Dataset", "Classifier", "Data Mode"]
    ].apply(lambda x: " ".join(x), axis=1)
    onehot_mixed_incomp_ids = onehot_mixed_incomplete[
        ["Dataset", "Classifier", "Data Mode"]
    ].apply(lambda x: " ".join(x), axis=1)
    to_concat.append(
        onehot_mixed_report[~onehot_mixed_rep_ids.isin(onehot_mixed_incomp_ids)]
    )
    print("got onehot -> mixed...")
    """

    df = pd.concat(to_concat)

    # fill N/A with the original parse mode if empty
    df.fillna("N/A", inplace=True)
    no_post_mode = df["Post Data Parse Mode"] == "N/A"
    df.loc[no_post_mode, "Post Data Parse Mode"] = df.loc[
        no_post_mode, "Data Parse Mode"
    ]

    hydra.core.global_hydra.GlobalHydra.instance().clear()
    return df


if __name__ == "__main__":
    df = load_df()
    # df.to_csv("data_mode_switch_results.csv", index=False)
    df.to_csv(f"{REPORTS_DIR}_results.csv", index=False)
    # print("Finished gathering results. Now computing regret.")
    # df_w_reg = compute_regret(df)
    # df_w_reg.to_csv("data_mode_switch_results_w_reg.csv", index=False)
