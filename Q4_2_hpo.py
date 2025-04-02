import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from functools import partial
import scikit_posthocs as sp
from matplotlib import colormaps
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from pathlib import Path


datadir = Path("./hpo-measure")
draw_settings = [("original", "-", "^"), ("distill", "--", "X")]
cmap = colormaps["tab20"]
TOP_N = 6
runs = []
for f in datadir.glob("*"):
    clf, dset = f.name.split("-")
    ori_search_log = pd.read_json(f / "original/tuned_search_log_000.json")
    dis_search_log = pd.read_json(f / "distill/tuned_search_log_000.json")
    hpo_reg = ori_search_log["value"].max() - dis_search_log["value"].max()
    runs.append(
        {
            "hpo_reg": hpo_reg,
            "dir": str(f),
            "Classifier": clf,
            "Dataset": dset,
        }
    )
all_runs = pd.DataFrame(runs).sort_values("hpo_reg").reset_index(drop=True)

for clf, runs in all_runs.groupby("Classifier"):
    # fig, ax = plt.subplots(figsize=(6, 3))
    fig, axes = plt.subplots(
        nrows=2,
        ncols=3,
        figsize=(6, 3),
        layout="constrained",
        # sharex=True,
        # sharey=True,
    )
    for i, f in enumerate(runs["dir"].iloc[:TOP_N]):
        dset = runs["Dataset"].iloc[i]
        ax = axes[i // 3, i % 3]
        for j, (k, linestyle, marker) in enumerate(draw_settings):
            f = Path(f)
            report = pd.read_json(f / k / "tuned_report_000.json")
            runtime = pd.read_json(f / k / "tuned_runtime_000.json")
            search_log = pd.read_json(f / k / "tuned_search_log_000.json")
            indiv_rt = np.array(
                runtime[runtime["Operation"] == "Train -- Hyperopt Per Run"][
                    "Time"
                ].values[0]
            )
            val_scos = np.maximum.accumulate(search_log["value"].values)
            if k == "original":
                dist_runtime = pd.read_json(f /  "distill/tuned_runtime_000.json")
                dist_all_rt = np.sum(
                    dist_runtime[dist_runtime["Operation"] == "Train -- Hyperopt Per Run"][
                        "Time"
                    ].values[0]
                )
                cutoff =np.cumsum(indiv_rt) < dist_all_rt*2 
                val_scos = val_scos[cutoff]
                indiv_rt = indiv_rt[cutoff]
            print(k, indiv_rt.sum())
            # if k == "distill":
            #     # we did 1000 runs for distillation, so let's take very 10 and group them.
            #     val_scos = np.bincount(np.arange(1000) // 10, val_scos) / 10
            #     indiv_rt = np.bincount(np.arange(1000) // 10, indiv_rt)
            ax.plot(
                np.cumsum(indiv_rt),
                val_scos,
                label=k,
                linestyle=linestyle,
                color=cmap(i * 2 + j),
                # marker=marker,
                # markerfacecolor=cmap(i*2 + j),
                # markeredgecolor="black",
                # markevery=[-1],
                # alpha=0.8,
                zorder=1,
            )
            # ax.scatter(
            #     np.cumsum(indiv_rt)[-1],
            #     val_scos[-1],
            #     # label=k,
            #     color=cmap(i * 2 + j),
            #     marker=marker,
            #     # markerfacecolor=cmap(i*2 + j),
            #     edgecolor="black",
            #     zorder=10,
            #     # alpha=0.8,
            # )
            best_idx = val_scos.argmax()
            # ax.scatter(
            #     np.cumsum(indiv_rt)[best_idx],
            #     val_scos[best_idx],
            #     # label=k,
            #     color=cmap(i * 2 + j),
            #     marker=marker,
            #     # markerfacecolor=cmap(i*2 + j),
            #     edgecolor="slategray",
            #     zorder=10,
            #     # alpha=0.8,
            # )
            ax.set_title(f"{dset}")
    # ax.set_xlabel("Time (s)")
    # ax.set_ylabel("Validation Score")
    # ax.set_ylim(0.65, 1.05)
    # ax.set_title(f"Top {TOP_N} runs for {clf}")
    ax.set_xscale("log")
    # ax.set_yscale("log")
    markers = [
        Line2D(
            [0,0.5],
            [0,0],
            # marker=ma,
            color="black",
            label="Full",
            markerfacecolor="slategray",
            linestyle="-",
            markersize=7,
        ),
        Line2D(
            [0,0.5],
            [0,0],
            # marker=ma,
            color="slategray",
            label="Distill",
            markerfacecolor="slategray",
            linestyle="--",
            markersize=7,
        )
    ]
    fig.supxlabel("Time (s)")
    fig.supylabel("Balanced Accuracy")
    ax = axes[0,-1]
    if "XGB" in clf:
        legend_m = ax.legend(
            handles=markers,
            loc="lower right",
            # title="Runs",
            bbox_to_anchor=(1.00, -0.06),
        )
    else:
        legend_m = ax.legend(
            handles=markers,
            loc="lower right",
            # title="Runs",
            bbox_to_anchor=(1.00, 0.02),
        )
    # ax.add_artist(legend_m)
    # fig.tight_layout()
    fig.savefig(
        f"./iclr-figures/rq4_2-hpo-runtimes-{clf}.pdf",
        bbox_inches="tight",
        # bbox_extra_artists=[legend_m],
    )

time_reduc = []
perf_reduc = []

for (dset,clf), runs in all_runs.groupby(["Dataset", "Classifier"]):
    runtime = pd.read_json(f /  "distill/tuned_runtime_000.json")
    indiv_rt = np.array(
        runtime[runtime["Operation"] == "Train -- Hyperopt Per Run"][
            "Time"
        ].values[0]
    )
    distill_runtime = indiv_rt.sum()

    runtime = pd.read_json(f /  "original/tuned_runtime_000.json")
    indiv_rt = np.array(
        runtime[runtime["Operation"] == "Train -- Hyperopt Per Run"][
            "Time"
        ].values[0]
    )
    original_runtime = indiv_rt.sum()

    print(f"{dset} {clf} {distill_runtime:.2f} {original_runtime:.2f} -- {distill_runtime/original_runtime * 100:.2f}%")



    report = pd.read_json(f / "distill/tuned_report_000.json")
    distill_sco = report[report["Subset"] == "Test"]["Score"].values[0]
    report = pd.read_json(f / "original/tuned_report_000.json")
    origianl_sco = report[report["Subset"] == "Test"]["Score"].values[0]

    time_reduc.append(distill_runtime/original_runtime)
    perf_reduc.append(distill_sco/origianl_sco)

time_reduc = np.array(time_reduc)
perf_reduc = np.array(perf_reduc)

print(f"Time reduction: {time_reduc.mean()*100:.2f}%, Perf match: {perf_reduc.mean()*100:.2f}%")
