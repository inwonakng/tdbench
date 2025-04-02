import hydra

from tabdd.results.load import load_all_enc_stats

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

hydra.core.global_hydra.GlobalHydra.instance().clear()

enc_stats = load_all_enc_stats(onehot_config)
enc_stats.to_csv("enc_stats.csv", index=False)
