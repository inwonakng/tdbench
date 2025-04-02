import pandas as pd
import json

dataset_names = {
    "Adult": "adult",
    "AmazonEmployeeAccess": "amazon_employee_access",
    "BankMarketing": "bank_marketing",
    "Credit": "credit",
    "CreditDefault": "credit_default",
    "Diabetes": "diabetes",
    "Electricity": "electricity",
    "Elevators": "elevators",
    "Higgs": "higgs",
    "HomeEquityCredit": "home_equity_credit",
    "House": "house",
    "Jannis": "jannis",
    "LawSchoolAdmissions": "law_school_admissions",
    "MagicTelescope": "magic_telescope",
    "MedicalAppointments": "medical_appointments",
    "MiniBooNE": "mini_boo_ne",
    "NumerAI": "numer_ai",
    "Nursery": "nursery",
    "PhishingWebsites": "phishing_websites",
    "Pol": "pol",
    "RoadSafety": "road_safety",
    "TencentCTRSmall": "tencent_ctr_small",
    "TwoDPlanes": "two_d_planes",
}

distill_methods = [
    "Random Sample",
    "<i>k</i>-means",
    "Agglomerative",
    "GM",
    "KIP",
]

dd_colors = [
    "rgba(99, 110, 250, 1.0)",
    "rgba(239, 85, 59, 1.0)",
    "rgba(0, 204, 150, 1.0)",
    "rgba(171, 99, 250, 1.0)",
    "rgba(255, 162, 90, 1.0)",
    "rgba(25, 210, 243, 1.0)",
    "rgba(255, 102, 146, 1.0)",
    "rgba(182, 232, 128, 1.0)",
    "rgba(255, 151, 255, 1.0)",
    "rgba(254, 202, 82, 1.0)",
]

encoders = [
    "GNN",
    "GNN-SFT",
    "MLP",
    "MLP-SFT",
    "TF",
    "TF-SFT",
]

enc_colors = [
    "rgba(228,26,28,1.0)",
    "rgba(55,126,184,1.0)",
    "rgba(77,175,74,1.0)",
    "rgba(152,78,163,1.0)",
    "rgba(255,127,0,1.0)",
    "rgba(255,255,51,1.0)",
    "rgba(166,86,40,1.0)",
    "rgba(247,129,191,1.0)",
    "rgba(153,153,153,1.0)",
]


def short_clf_name(clf):
    if clf == "XGBClassifier":
        return "XGB"
    elif clf == "MLPClassifier":
        return "MLP"
    elif clf == "LogisticRegression":
        return "LR"
    elif clf == "GaussianNB":
        return "NB"
    elif clf == "KNeighborsClassifier":
        return "KNN"
    else:
        return clf


# helper funciton to parse large groups. Sometimes we only want to know the best performance
# in the group instead of the mean.
def get_group_aggr(aspects, direction: str = "max"):
    _aggr = max if direction == "max" else min

    def aggr_func(group, metric):
        return _aggr(gr[metric].mean() for _, gr in group.groupby(aspects))

    return aggr_func


def describe_df(rankings):
    ranks_min = rankings.min()
    ranks_mean = rankings.mean()
    ranks_std = rankings.std()
    ranks_median = rankings.median()
    ranks_max = rankings.max()
    ranks_q_1 = rankings.quantile(0.25)
    ranks_q_2 = rankings.quantile(0.50)
    ranks_q_3 = rankings.quantile(0.75)

    stats = pd.DataFrame(
        {
            "min": ranks_min,
            "mean": ranks_mean,
            "std": ranks_std,
            "median": ranks_median,
            "max": ranks_max,
            "25%": ranks_q_1,
            "50%": ranks_q_2,
            "75%": ranks_q_3,
        }
    ).sort_values("mean")
    return stats


def format_cell(value):
    if isinstance(value, float):
        if value % 1 == 0:
            return str(int(value))
        return f"{value:.4f}"
    return str(value)


def save_table_json(table, path):
    with path.open("w") as f:
        json.dump(
            {
                "columns": [""] + table.columns.tolist(),
                "values": [
                    [idx, *[format_cell(cell) for cell in row]]
                    for idx, row in zip(table.index, table.values.tolist())
                ],
            },
            f,
        )
