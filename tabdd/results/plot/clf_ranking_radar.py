import hydra
import pandas as pd
import matplotlib.pyplot as plt

from tabdd.config.paths import RUN_CONFIG_DIR, FIGURES_DIR
from tabdd.results.load import load_all_clf_perf, load_all_enc_stats, compute_rank
from tabdd.results.plot.matplotlib import make_radarplot, radar_factory


def get_dm_ranks_per_enc(
    report: pd.DataFrame,
    encoder: str,
) -> tuple[list[str], list[str], list[list[float]]]:
    if encoder == "all":
        df = report
    else:
        df = report[report["Encoder"] == encoder]
    score_rank = compute_rank(df, target="Distill Method", metric="Score")
    opt_tr_time_rank = compute_rank(
        df, target="Distill Method", metric="Opt Train Time Total", direction="min"
    )
    def_tr_time_rank = compute_rank(
        df, target="Distill Method", metric="Default Train Time", direction="min"
    )
    te_time_rank = compute_rank(
        df, target="Distill Method", metric="Inference Time", direction="min"
    )

    distill_methods = list(
        set(report["Distill Method"].unique())
        - set(["Original", "Mixed Original", "Encoded", "Decoded"])
    )

    ranks = pd.DataFrame(
        [
            {
                "Criteria": "Score",
                **{
                    dm: score_rank[score_rank["Distill Method"] == dm]["Rank"].mean()
                    for dm in distill_methods
                },
            },
            {
                "Criteria": "Opt Train Time",
                **{
                    dm: opt_tr_time_rank[opt_tr_time_rank["Distill Method"] == dm][
                        "Rank"
                    ].mean()
                    for dm in distill_methods
                },
            },
            {
                "Criteria": "Default Train Time",
                **{
                    dm: def_tr_time_rank[def_tr_time_rank["Distill Method"] == dm][
                        "Rank"
                    ].mean()
                    for dm in distill_methods
                },
            },
            {
                "Criteria": "Inference Time",
                **{
                    dm: te_time_rank[te_time_rank["Distill Method"] == dm][
                        "Rank"
                    ].mean()
                    for dm in distill_methods
                },
            },
        ]
    )

    ranks.iloc[:, 1:] = len(distill_methods) - ranks.iloc[:, 1:]

    return ranks


def dm_compare_plot(
    report: pd.DataFrame,
):

    overall_ranks = get_dm_ranks_per_enc(report, "all")
    no_enc_ranks = get_dm_ranks_per_enc(report, "")

    radar_factory(len(overall_ranks))
    fig, axes = plt.subplots(
        nrows=2, ncols=4, subplot_kw=dict(projection="radar"), figsize=(16, 7)
    )
    fig.subplots_adjust(wspace=0.4, hspace=0.20, top=1.0, bottom=0.05)

    print("Overall Ranks")
    print(overall_ranks)
    print()

    make_radarplot(
        data=overall_ranks.values[:, 1:].T,
        variables=overall_ranks.values[:, 0],
        ax=axes[0, 0],
    )
    axes[0, 0].set_title("Overall")

    print("No Encoder Ranks")
    print(no_enc_ranks)
    print()

    make_radarplot(
        data=no_enc_ranks.values[:, 1:].T,
        variables=no_enc_ranks.values[:, 0],
        ax=axes[1, 0],
    )
    axes[1, 0].set_title("No Encoder")

    for i, enc in enumerate(["MLP", "GNN", "TF"]):
        base_enc_ranks = get_dm_ranks_per_enc(report, enc)
        multi_enc_ranks = get_dm_ranks_per_enc(report, f"{enc}-MultiHead")

        enc_pretty_name = enc
        if enc == "MLP":
            enc_pretty_name = "FFN"

        print(f"{enc_pretty_name} Ranks")
        print(base_enc_ranks)
        print()

        make_radarplot(
            data=base_enc_ranks.values[:, 1:].T,
            variables=base_enc_ranks.values[:, 0],
            ax=axes[0, i + 1],
        )
        axes[0, i + 1].set_title(enc_pretty_name)

        print(f"{enc_pretty_name} FT Ranks")
        print(multi_enc_ranks)
        print()

        make_radarplot(
            data=multi_enc_ranks.values[:, 1:].T,
            variables=multi_enc_ranks.values[:, 0],
            ax=axes[1, i + 1],
        )
        axes[1, i + 1].set_title(f"{enc_pretty_name} FT")
        break

    legend = axes[0, -1].legend(
        set(overall_ranks.columns) - set(["Criteria"]),
        loc=(0.9, 0.95),
        labelspacing=0.1,
        fontsize="small",
    )
    fig.savefig(FIGURES_DIR / "dm_compare_radar.pdf", bbox_inches="tight")


@hydra.main(version_base=None, config_path=RUN_CONFIG_DIR, config_name="all")
def run(config):
    report, incomplete = load_all_clf_perf(config)
    dm_compare_plot(report)


if __name__ == "__main__":
    run()
