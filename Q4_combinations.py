import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import colormaps
from functools import partial
import scikit_posthocs as sp


# df = pd.read_csv("./parsed_with_rel_reg.csv")


df = pd.read_csv("./data_mode_switch_results.csv", low_memory=False)
# df_mix_tf = pd.read_csv("./mixed_tf_results.csv")
# df = pd.concat([df, df_mix_tf])
df = df[df["Subset"] == "Test"]
df["Use Encoder"] = df["Encoder"].notna()
df = df[df["Classifier"] != "GaussianNB"]
df["Encoder"] = df["Encoder"].replace(
    {
        "MLP": "FFN",
        "TF-MultiHead": "TF*",
        "GNN-MultiHead": "GNN*",
        "MLP-MultiHead": "FFN*",
    }
)
df["Distill Method"] = df["Distill Method"].replace(
    {
        "KMeans": "KM",
        "Agglo": "AG",
    }
)


def mixed_distill(df, distill_method):
    return df[
        (df["Distill Method"] == distill_method)
        & (df["Data Parse Mode"] == "mixed")
        & (df["Post Data Parse Mode"] == "mixed")
        & (~df["Use Encoder"])
    ]


def encoded_distill_encoded(df, distill_method, encoder):
    return df[
        (df["Distill Method"] == distill_method)
        & (df["Data Parse Mode"] == "onehot")
        & (df["Post Data Parse Mode"] == "onehot")
        & (df["Output Space"] == "encoded")
        & (df["Encoder"] == encoder)
        & (df["Use Encoder"])
    ]


def encoded_distill_mixed(df, distill_method, encoder):
    return df[
        (df["Distill Method"] == distill_method)
        & (df["Data Parse Mode"] == "onehot")
        & (df["Post Data Parse Mode"] == "mixed")
        & (df["Output Space"].isin(["decoded", "original"]))
        & (df["Encoder"] == encoder)
        & (df["Use Encoder"])
    ]


def no_encode_distill(df, distill_method):
    return df[(df["Distill Method"] == distill_method) & (~df["Use Encoder"])]


cmap = colormaps["tab20c"]

encoders = [
    "TF",
    "GNN",
    "FFN",
    "TF*",
    "GNN*",
    "FFN*",
]

distill_methods = [
    "KM",
    "AG",
    "KIP",
    "GM",
]

METHOD_GROUPS = {
    "RND": {
        "filter": partial(mixed_distill, distill_method="Random Sample"),
        "color": "black",
        "linestyle": "-",
    },
}
for dm in distill_methods:
    for enc in encoders:
        METHOD_GROUPS[f"{dm} {enc} Enc."] = {
            "filter": partial(encoded_distill_encoded, distill_method=dm, encoder=enc),
        }
        METHOD_GROUPS[f"{dm} {enc} Recon."] = {
            "filter": partial(encoded_distill_mixed, distill_method=dm, encoder=enc),
        }

to_cat = []
for (dset, clf), by_dset_clf in df.groupby(
    [
        "Dataset",
        "Classifier",
    ]
):
    ori_sco = by_dset_clf[
        (by_dset_clf["Distill Method"] == "Original")
        & (by_dset_clf["Data Parse Mode"] == "mixed")
        & (by_dset_clf["Post Data Parse Mode"] == "mixed")
    ]["Score"].mean()
    rnd_sco = by_dset_clf[
        (by_dset_clf["Distill Method"] == "Random Sample")
        & (by_dset_clf["Data Parse Mode"] == "mixed")
        & (by_dset_clf["Post Data Parse Mode"] == "mixed")
        & (by_dset_clf["N"] == 10)
    ]["Score"].mean()
    if rnd_sco >= ori_sco:
        print(f"{clf}:{dset} Skipped.. Too easy")
        continue
    for k, grp in METHOD_GROUPS.items():
        filtered = grp["filter"](by_dset_clf)
        if filtered.empty:
            print(f"{clf}:{dset} -- {k} Skipped.. No data")
            continue
        data = [
            (n, (ori_sco - by_n_sn["Score"].mean()) / (ori_sco - rnd_sco))
            for (n, sn), by_n_sn in filtered.groupby(["N", "Short Name"])
        ]
        if len(data) == 0:
            print(f"{clf}:{dset} -- {k} Skipped.. No data")
            continue
        best_scos = pd.DataFrame(
            [
                (n, by_n["Regret"].min())
                for n, by_n in pd.DataFrame(data, columns=["N", "Regret"]).groupby("N")
            ],
            columns=["N", "Regret"],
        )
        best_scos["Method"] = k
        best_scos["Dataset"] = dset
        best_scos["Classifier"] = clf
        to_cat.append(best_scos)
finalized = pd.concat(to_cat)

clf_comp = {}
for clf, by_clf in finalized.groupby("Classifier"):
    min_dsets = set(by_clf["Dataset"].unique())
    fair_comp = {}
    for k, grp in METHOD_GROUPS.items():
        scores = by_clf[by_clf["Method"] == k]
        min_dsets &= set(scores["Dataset"].unique())
        fair_comp[k] = scores
    print("-" * 40)
    print(f"{clf} with {len(min_dsets)} datasets")
    print("=" * 40 + "\n")
    fair_comp = {k: v[v["Dataset"].isin(min_dsets)] for k, v in fair_comp.items()}
    clf_comp[clf] = fair_comp
finalized2 = pd.concat([vv for v in clf_comp.values() for vv in v.values()])
finalized2["Rank"] = finalized2.groupby(["Dataset", "Classifier", "N"])["Regret"].rank()

"""
Processing complete
"""



"""
Table of comparison through all N
"""


def method_to_components(method):
    parts = method.split(" ")
    components = {
        "Encoder": "N/A",
        "D.M.": "N/A",
        "Output": "N/A",
    }
    if len(parts) == 1:
        components["D.M."] = parts[0]
    elif len(parts) == 3:
        components["D.M."] = parts[0]
        components["Encoder"] = parts[1]
        if parts[2] == "Enc.":
            components["Output"] = "Enc."
        elif parts[2] == "Recon.":
            components["Output"] = "Recon."
    return components

top10 = finalized2[finalized2["Rank"] <= 10]
table = pd.DataFrame(
    [
        {
            "Count": int(count),
            **method_to_components(method),
        }
        for method, count in top10["Method"].value_counts()[:5].items()
    ]
)
print(table.to_markdown(index=False))
with open("./iclr-figures/rq4-top-count.tex", "w") as f:
    f.write(table.to_latex(index=False))

for n in [10, 50, 100]:
    filtered = finalized2[finalized2["N"] == n]
    top10 = filtered[filtered["Rank"] <= 10]
    table = pd.DataFrame(
        [
            {
                "Count": int(count),
                **method_to_components(method),
            }
            for method, count in top10["Method"].value_counts()[:5].items()
        ]
    )
    print(table.to_markdown(index=False))
    with open(f"./iclr-figures/rq4-top-count-n{n}.tex", "w") as f:
        f.write(table.to_latex(index=False))
