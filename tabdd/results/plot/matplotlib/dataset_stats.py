import hydra
from omegaconf import DictConfig
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

from tabdd.results.load import load_all_dataset_stats
from tabdd.config.paths import RUN_CONFIG_DIR, FIGURES_DIR


def plot_ds_stats(
    config: DictConfig,
    normalize: bool = False,
):
    ds_stats = load_all_dataset_stats(config)

    max_label_count = ds_stats["Label Ratio"].apply(len).max()

    for i in range(max_label_count):
        ds_stats[f"label_{i}"] = ds_stats["Label Ratio"].apply(lambda x: x[i])

    # Reshape DataFrame from wide to long format
    fig, axes = plt.subplots(figsize=(8, 5), ncols=2, sharey=True)

    sns.barplot(
        x=(np.ones(len(ds_stats["Dataset"].unique())) if normalize else "Rows"),
        y="Dataset",
        data=ds_stats,
        orient="h",
        ax=axes[0],
        label="0",
    )

    sns.barplot(
        x=(ds_stats["label_1"] / ds_stats["Rows"] if normalize else "label_1"),
        y="Dataset",
        data=ds_stats,
        orient="h",
        ax=axes[0],
        label="1",
    )
    h, l = axes[0].get_legend_handles_labels()
    axes[0].legend(h[::-1], ["0", "1"], title="Label", bbox_to_anchor=(1, 1))
    axes[0].set_xlabel("Label Distribution")

    sns.barplot(
        x=(
            np.ones(len(ds_stats["Dataset"].unique()))
            if normalize
            else "Original Features"
        ),
        y="Dataset",
        data=ds_stats,
        orient="h",
        ax=axes[1],
        color=sns.color_palette("tab10")[2],
        label="Continuous",
    )

    sns.barplot(
        x=(
            ds_stats["Categorical Features"] / ds_stats["Original Features"]
            if normalize
            else "Categorical Features"
        ),
        y="Dataset",
        data=ds_stats,
        orient="h",
        ax=axes[1],
        color=sns.color_palette("tab10")[3],
        label="Categorical",
    )

    h, l = axes[1].get_legend_handles_labels()
    axes[1].legend(
        h[::-1],
        [
            "Categorical",
            "Continuous",
        ],
        title="Feature Kind",
        bbox_to_anchor=(1, 1),
    )
    axes[1].set_xlabel("Feature Kind Distribution")

    fig.tight_layout()

    filename = "dataset_stats.pdf"
    if normalize:
        filename = "dataset_stats_normalized.pdf"
    fig.savefig(FIGURES_DIR / filename, bbox_inches="tight")


@hydra.main(version_base=None, config_path=RUN_CONFIG_DIR, config_name="tune")
def run(config: DictConfig) -> None:
    plot_ds_stats(config, normalize=False)
    plot_ds_stats(config, normalize=True)


if __name__ == "__main__":
    run()
