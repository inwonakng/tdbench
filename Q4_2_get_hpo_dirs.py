import pandas as pd
import subprocess
from pathlib import Path


df = pd.read_csv("./hpo_measure_results.csv", low_memory=False)
# df_mix_tf = pd.read_csv("./mixed_tf_results.csv")
# df = pd.concat([df, df_mix_tf])
df = df[df["Subset"] == "Test"]
df["Use Encoder"] = df["Encoder"].notna()
df = df[df["Classifier"] != "GaussianNB"]

dp_mapping = {
    "mixed": "mixed/standard/uniform-10",
    "onehot": "onehot/standard/uniform-10",
}


def parse_enc_name(encname):
    if "-MultiHead" in encname:
        return "multihead" + encname.split("-MultiHead")[0].lower() + "autoencoder"
    else:
        return encname.lower() + "autoencoder"


def get_cache_dir(runinfo):
    clf = runinfo["Classifier"]
    dset = runinfo["Dataset"]
    dp = runinfo["Data Parse Mode"]
    dm = runinfo["Distill Method"]
    dspace = runinfo["Distill Space"]
    ospace = runinfo["Output Space"]
    cvt_bin = runinfo["Convert Binary"]
    cls_center = runinfo["Cluster Center"]
    enc = runinfo["Encoder"]
    post_proc = runinfo["Post Data Parse Mode"]
    n = runinfo["N"]
    cachedir = (
        f"{clf.lower()}/{dset}/{dp_mapping[dp]}/hyperopt/balanced_accuracy/0_folds/"
    )
    if dm == "Original":
        cachedir += "original/"
    elif dm == "KMeans":
        cachedir += f"kmeans_{dspace}_{ospace}"
        if cvt_bin:
            cachedir += "_binary"
        cachedir += f"/{cls_center}/"
        if post_proc != dp:
            cachedir += f"post_process/{dp_mapping[post_proc]}/"
        cachedir += f"{parse_enc_name(enc)}/16/N={n}/"
    else:
        raise NotImplementedError(f"Distill method [{dm}] not found")
    return cachedir


cols_of_interest = [
    "Distill Method",
    "Short Name",
    "Score",
    "Default Train Time",
    "Opt Train Time",
    "Opt Train Time Total",
]

pairs = []

for (clf, dset), grp in df.groupby(["Classifier", "Dataset"]):
    ori_grp = grp[grp["Distill Method"] == "Original"]
    if ori_grp.empty:
        continue
    ori_setup = ori_grp.iloc[0]
    ori_sco = ori_setup["Score"]
    distill_grp = grp[grp["Distill Method"] != "Original"]
    if distill_grp.empty:
        continue
    best_distill = grp[grp["Distill Method"] != "Original"]["Score"].max()
    best_setup = distill_grp[distill_grp["Score"] == best_distill].iloc[0]
    print("=" * 20)
    print(clf, dset)
    print("=" * 20)
    print(ori_setup[cols_of_interest])
    print()
    print("-" * 20 + "\n")
    print(best_setup[cols_of_interest])
    print()
    print("=" * 20 + "\n")

    Path(f"./hpo-measure/{clf}-{dset}/").mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "scp",
            "-rp",
            f"silkworm:~/tabdd-data/hpo_measure/{get_cache_dir(ori_setup)}",
            f"./hpo-measure/{clf}-{dset}/original",
        ]
    )

    subprocess.run(
        [
            "scp",
            "-rp",
            f"silkworm:~/tabdd-data/hpo_measure/{get_cache_dir(best_setup)}",
            f"./hpo-measure/{clf}-{dset}/distill",
        ]
    )
