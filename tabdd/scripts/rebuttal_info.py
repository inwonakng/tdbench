import pandas as pd

df = pd.read_csv("data_mode_switch_results_w_reg.csv")
df = df[df["Subset"] == "Test"]

# just redo regret

w_reg = []
for (ds, clf, n), grouped in df.groupby(["Dataset", "Classifier", "N"]):
    ori = grouped[
        (grouped["Data Parse Mode"] == "mixed")
        & (grouped["Distill Method"] == "Original")
    ]
    grouped["Regret"] = (ori["Score"] - grouped["Score"]) / ori["Score"]
    w_reg.append(grouped)
w_reg = pd.concat(w_reg)

mixed = w_reg[w_reg["Data Parse Mode"] == "mixed"]
onehot = w_reg[w_reg["Data Parse Mode"] == "onehot"]

# first onehot mode detailed regret score

print(
    df[df["Distill Method"] != "Original"]
    .groupby(["Distill Method", "Classifier"])["Regret"]
    .mean()
    .reset_index()
    .pivot_table(index="Distill Method", columns="Classifier", values="Regret")
    .to_markdown(floatfmt=".4f")
)

# now to check if KIP without encoder is indeed better
df[(df["Distill Method"] == "KIP")].groupby("Encoder")["Regret"].mean().reset_index()

print(
    df[df["Encoder"].isna()]
    .groupby(["Distill Method", "N"])["Regret"]
    .mean()
    .reset_index()
    .pivot_table(index="Distill Method", columns="N", values="Regret")
    .to_markdown(floatfmt=".4f")
)

enc_compare = []
for enc, by_enc in df.groupby("Encoder", dropna=False):
    if pd.isna(enc):
        enc_name = "No Encoder"
    else:
        enc_name = enc
    for dm, by_dm in by_enc[
        ~by_enc["Distill Method"].isin(["Original", "Random Sample"])
        & ~by_enc["Output Space"].isin(["decoded"])
    ].groupby("Distill Method"):
        m = by_dm["Regret"].mean()
        s = by_dm["Regret"].std()
        enc_compare.append(
            {
                # "Regret": f"{m:.4f} ± {s:.4f}",
                "Regret": f"{m:.4f}",
                "Encoder": enc_name,
                "Distill Method": dm,
            }
        )
enc_compare = pd.DataFrame(enc_compare)
print(
    enc_compare.pivot_table(
        index="Encoder",
        columns="Distill Method",
        values="Regret",
        aggfunc=lambda x: " ".join(x),
    ).to_markdown()
)

# now comapre col binary and not
df[
    (df["Distill Method"] == "Original")
    & (df["N"] == 100)
    & (df["Classifier"] == "XGBClassifier")
    & (df["Dataset"] == "Adult")
    & (df["Data Parse Mode"] == "onehot")
]


compare_bin = []
for (clf, ds), by_clf in df[
    (df["Distill Method"] == "Original") & (df["N"] == 100)
].groupby(["Classifier", "Dataset"]):
    mix_score = by_clf[by_clf["Data Parse Mode"] == "mixed"]["Score"].values.min()
    oh_score = by_clf[by_clf["Data Parse Mode"] == "onehot"]["Score"].values.max()
    compare_bin.append(
        {
            "Classifier": clf,
            "Dataset": ds,
            "Original": mix_score,
            "Binary": oh_score,
        }
    )

compare_bin = pd.DataFrame(compare_bin)
print("Original vs Binary raw clf")
print(
    compare_bin.groupby("Classifier")[["Original", "Binary"]]
    .mean()
    .to_markdown(floatfmt=".4f")
)

compare_bin_clf = []
for (clf, ds, n), by_clf in df[
    (~df["Distill Method"].isin(["Original", "Random Sample"]))
    & (~df["Classifier"].isin(["ResNet", "FTTransformer"]))
].groupby(["Classifier", "Dataset", "N"]):
    mix_score = by_clf[by_clf["Data Parse Mode"] == "mixed"]["Regret"].values.min()
    oh_score = by_clf[by_clf["Data Parse Mode"] == "onehot"]["Regret"].values.min()
    compare_bin_clf.append(
        {
            "Classifier": clf,
            "Dataset": ds,
            "N": n,
            "Original": mix_score,
            "Binary": oh_score,
        }
    )
compare_bin_clf = pd.DataFrame(compare_bin_clf)

print("Original vs Binary in distill per classifier")
print(
    compare_bin_clf.groupby("Classifier")[["Original", "Binary"]]
    .mean()
    .to_markdown(floatfmt=".4f")
)

compare_bin_dm = []
for (dm, ds, n), by_clf in df[
    (~df["Distill Method"].isin(["Original", "Random Sample"]))
    & (~df["Classifier"].isin(["ResNet", "FTTransformer"]))
].groupby(["Distill Method", "Dataset", "N"]):
    mix_score = by_clf[by_clf["Data Parse Mode"] == "mixed"]["Regret"].values.min()
    oh_score = by_clf[by_clf["Data Parse Mode"] == "onehot"]["Regret"].values.min()
    compare_bin_dm.append(
        {
            "Distill Method": dm,
            "Dataset": ds,
            "N": n,
            "Original": mix_score,
            "Binary": oh_score,
        }
    )
compare_bin_dm = pd.DataFrame(compare_bin_dm)

print("Original vs Binary in classifier per distill")
print(
    compare_bin_dm.groupby("Distill Method")[["Original", "Binary"]]
    .mean()
    .to_markdown(floatfmt=".4f")
)

# runtimes
runtimes = pd.read_csv("runtime_results.csv")
runtimes = runtimes[
    (runtimes["Classifier"] == "XGBClassifier")
    & (runtimes["Dataset"].isin(["Adult", "BankMarketing", "PhishingWebsites"]))
    & (runtimes["Subset"] == "Test")
]

w_reg = []
for (ds, clf, n), grouped in runtimes.groupby(["Dataset", "Classifier", "N"]):
    ori = df[
        (df["Data Parse Mode"] == "mixed")
        & (df["Distill Method"] == "Original")
        & (df["Dataset"] == ds)
        & (df["Classifier"] == clf)
        & (df["N"] == n)
    ]
    grouped["Regret"] = (ori["Score"].values - grouped["Score"].values) / ori[
        "Score"
    ].values
    w_reg.append(grouped)
runtimes_w_reg = pd.concat(w_reg)

runtime_table = []
for (enc, clf, ds), by_mo in runtimes_w_reg[
    ~runtimes_w_reg["Distill Method"].isin(["Original"])
].groupby(["Encoder", "Classifier", "Dataset"], dropna=False):
    ori_time = runtimes_w_reg[
        (runtimes_w_reg["Distill Method"] == "Original")
        & (runtimes_w_reg["Classifier"] == clf)
        & (runtimes_w_reg["Dataset"] == ds)
        & (runtimes_w_reg["N"] == 10)
    ]["Default Train Time"].mean()
    runtime_table.append(
        {
            "Dataset": ds,
            "Distill Method": "Original",
            "Classifier": clf,
            "Encoder": enc,
            "N": 0,
            "Regret": 0,
            "Distill Time": 0,
            "Train Time": ori_time,
        }
    )
    for dm, by_dm in by_mo.groupby("Distill Method"):
        for n, by_n in by_dm.groupby("N"):
            runtime_table.append(
                {
                    "Dataset": ds,
                    "Distill Method": dm,
                    "Classifier": clf,
                    "Encoder": enc,
                    "N": n,
                    "Regret": by_n["Regret"].sort_values()[:5].mean(),
                    "Distill Time": (
                        by_n["Data Distill Time"].sort_values()[:5].mean()
                        if dm != "Random Sample"
                        else 0
                    ),
                    "Train Time": by_n["Default Train Time"].sort_values()[:5].mean(),
                }
            )
runtime_table = pd.DataFrame(runtime_table)

# now make table for each method

for (dm, enc), by_dm in runtime_table.groupby(
    ["Distill Method", "Encoder"], dropna=False
):
    if dm == "Original":
        continue
    if pd.isna(enc):
        enc_name = "No Encoder"
    else:
        enc_name = enc.replace("-MultiHead", "*")
    print(f"Distill Method: {dm}, Encoder: {enc_name}")
    print(
        by_dm.pivot_table(
            index="N",
            values=["Regret", "Distill Time", "Train Time"],
            aggfunc="mean",
        )[["Regret", "Distill Time", "Train Time"]].to_markdown(floatfmt=".4f")
    )
    print()


print("No Distill")
print(
    runtime_table[(runtime_table["Distill Method"] == "Original")]
    .groupby("Dataset")[["Regret", "Train Time"]]
    .mean()
    .to_markdown(floatfmt=".4f")
)

# now show which has the best benefit. Average over N

avgs = (
    runtime_table[
        ~(
            (runtime_table["Distill Method"] == "Original")
            & ~(runtime_table["Encoder"].isna())
        )
    ]
    .groupby(["Distill Method", "Encoder"], dropna=False)[
        ["Regret", "Distill Time", "Train Time"]
    ]
    .mean()
    .reset_index()
)
avgs["Total Time (s)"] = avgs[["Distill Time", "Train Time"]].sum(1)
avgs["Encoder"].str.replace("-MultiHead", "*")

avgs["Name"] = names = [
    dm + ("" if pd.isna(enc) else " + " + enc.replace("-MultiHead", "*"))
    for dm, enc in avgs[["Distill Method", "Encoder"]].values
]

print(avgs[["Name", "Regret", "Total Time (s)"]].pivot_table(
    index="Name", values=["Regret", "Total Time (s)"], aggfunc="mean"
).to_markdown(floatfmt=".4f"))



# import seaborn as sns
# import matplotlib.pyplot as plt
#
# fig,ax = plt.subplots(figsize=(4,4))
# s = sns.scatterplot(
#     data=avgs,
#     x = "Total Time (s)",
#     y = "Regret",
#     ax = ax,
#     hue="Name",
#     style="Name",
#     s=60,
# )
# s.set(xscale="log")
#
# sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
# fig.savefig("regret_vs_runtime.pdf", bbox_inches="tight")
