import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import colormaps
from functools import partial
from tqdm.auto import tqdm
import numpy as np

# df = pd.read_csv("./parsed_with_rel_reg.csv")


df = pd.read_csv("./data_mode_switch_results.csv", low_memory=False)
df = df[df["Subset"] == "Test"]

dm_rows =[ "N", "Dataset", "Data Parse Mode", "Post Data Parse Mode","Data Mode", "Distill Method", "Encoder", "Distill Space", "Output Space", "Convert Binary", "Cluster Center", "Short Name"] 
# b/c we have 5 random iteration for everything
uniq_dd = df[dm_rows].value_counts().shape[0] * 5
print(f"unique datasets: {uniq_dd}")

# b/c we have 5 random iteration for everything
uniq_clf = df[dm_rows+["Classifier"]].value_counts().shape[0] * 5
print(f"unique classifiers: {uniq_clf}")





vvs = [
    70,
    "onehot",
    "onehot",
    "GM",
    "GNN",
    "encoded",
    "decoded",
    True,
    "centroid",
]
mask = True
for k, v in zip(dm_rows, vvs):
    mask &= df[k] == v
df[mask]
