import pandas as pd
import numpy as np

rank_per_dm = pd.read_csv("./rank_per_dm.csv")
rank_per_dm = rank_per_dm[rank_per_dm["Cluster Center"] == "centroid"]
rank_per_enc = pd.read_csv("./rank_per_enc.csv")
rank_per_enc = rank_per_enc[rank_per_enc["Cluster Center"] == "centroid"]
rank_per_os = pd.read_csv("./rank_per_os.csv")
rank_per_os = rank_per_os[rank_per_os["Cluster Center"] == "centroid"]
rank_per_clf = pd.read_csv("./rank_per_clf.csv")
rank_per_clf = rank_per_clf[rank_per_clf["Cluster Center"] == "centroid"]

CLASSIFIERS = [
    "XGBClassifier",
    "LogisticRegression",
    "MLPClassifier",
    "KNeighborsClassifier",
    "GaussianNB",
]
ENCODERS = [
    "MLP",
    "MLP-MultiHead",
    "GNN",
    "GNN-MultiHead",
]
DISTILL_METHODS = [
    "KMeans",
    "Agglo",
    "KIP",
]


def short_dm_name(dm):
    if dm == "Random Sample":
        return "RS"
    elif dm == "KMeans":
        return "KM"
    elif dm == "Agglo":
        return "AG"
    else:
        return dm


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


def short_enc_name(enc):
    return enc.replace("MLP", "FFN").replace("MultiHead", "FT")


def pretty_name(kind, value):
    if kind == "Distill Method":
        return short_dm_name(value)
    elif kind == "Encoder":
        return short_enc_name(value)
    elif kind == "Classifier":
        return short_clf_name(value)
    else:
        return value


def to_latex_table(
    df: pd.DataFrame, caption: str = "Caption", label: str = "my_label"
) -> str:
    header = "\t\t" + " & ".join([f"\\textbf{{{c}}}" for c in df.columns])
    layout = "c" * len(df.columns)
    body = "\n".join(
        "\t\t" + " & ".join(str(cell) for cell in row) + "\\\\" for row in df.values
    )
    table = ""
    table += f"\\begin{{table*}}[]\n"
    table += f"\t\\centering\n"
    table += f"\t{{\\small\n"
    table += f"\t\\begin{{tabular}}{{{layout}}}\n"
    table += f"\t\t\\toprule\n"
    table += f"{header}\\\\ \\midrule\n"
    table += f"{body}\n"
    table += f"\t\t\\bottomrule\n"
    table += f"\t\\end{{tabular}}\n"
    table += f"\t}}\n"
    table += f"\t\\caption{{{caption}}}\n"
    table += f"\t\\label{{{label}}}\n"
    table += f"\\end{{table*}}\n"
    return table


def make_table(
    df: pd.DataFrame,
    target_col: str,
    group1: dict[str, list[any]],
    group2: dict[str, list[any]],
    value_col: str,
    cell_parser: callable,
) -> str:
    table = []
    for g1 in group1["values"]:
        for g2 in group2["values"]:
            one_row = {
                group1["name"]: pretty_name(group1["name"], g1),
                group2["name"]: pretty_name(group2["name"], g2),
            }
            for target, each_target in df[
                (df[group1["name"]] == g1) & (df[group2["name"]] == g2)
            ].groupby(target_col):
                target_name = pretty_name(target_col, target)
                target_values = each_target[value_col]
                one_row[target_name] = cell_parser(target_values)
                print(target_name, one_row[target_name])
            print(one_row)
            table += [one_row]
    table = pd.DataFrame(table).rename(columns={"N": "$n/L$"})
    print(table)
    print(table.transpose())
    # return table
    return to_latex_table(table)


def mean_std(values):
    return f"{np.mean(values):.2f} $\pm$ {np.std(values):.2f}"


def mean(values):
    return f"{np.mean(values):.2f}"


# Rank per os by n, dm
table = make_table(
    df=rank_per_os,
    target_col="Output+Encoder Space",
    group1={"name": "N", "values": [10, 50, 100]},
    group2={"name": "Distill Method", "values": ["KMeans", "Agglo", "KIP"]},
    value_col="Rank",
    cell_parser=mean_std,
)

print("Rank per OS by N, DM")
print(table)
quit()
open("figures/rank_per_os_dm_N.tex", "w").write(table)

# Rank per enc by n, dm
table = make_table(
    df=rank_per_enc,
    target_col="Encoder",
    group1={"name": "N", "values": [10, 50, 100]},
    group2={"name": "Distill Method", "values": ["KMeans", "Agglo", "KIP"]},
    value_col="Rank",
    cell_parser=mean_std,
)

print("Rank per ENC by N, DM")
open("figures/rank_per_enc_dm_N.tex", "w").write(table)

# Rank per dm by n, enc
table = make_table(
    df=rank_per_dm,
    target_col="Distill Method",
    group1={"name": "N", "values": [10, 50, 100]},
    group2={"name": "Encoder", "values": ENCODERS},
    value_col="Rank",
    cell_parser=mean_std,
)

print("Rank per DM by ENC, DM")
open("figures/rank_per_dm_enc_N.tex", "w").write(table)

# Rank per clf by n, dm
table = make_table(
    df=rank_per_clf,
    target_col="Classifier",
    group1={"name": "N", "values": [10, 50, 100]},
    group2={
        "name": "Distill Method",
        "values": ["Random Sample", "KMeans", "Agglo", "KIP"],
    },
    value_col="Rank",
    cell_parser=mean_std,
)

print("Rank per CLF by N, DM")
open("figures/rank_per_clf_dm_N.tex", "w").write(table)

# Rank per dm by n, enc for logreg
table = make_table(
    df=rank_per_dm[rank_per_dm["Classifier"] == "LogisticRegression"],
    target_col="Distill Method",
    group1={"name": "N", "values": [10, 50, 100]},
    group2={
        "name": "Encoder",
        "values": ENCODERS,
    },
    value_col="Rank",
    cell_parser=mean_std,
)

print("Rank per DM by N, ENC for LogReg")
open("figures/rank_per_dm_enc_N_logreg.tex", "w").write(table)
