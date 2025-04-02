import hydra
import pandas as pd
import matplotlib.pyplot as plt

from tabdd.config.paths import RUN_CONFIG_DIR, FIGURES_DIR
from tabdd.results.load import load_all_clf_perf, load_all_enc_stats, compute_rank
from tabdd.results.plot.matplotlib import make_radarplot, radar_factory


def get_enc_ranks(
    report: pd.DataFrame,
    enc_stats: pd.DataFrame,
    multihead: bool = False,
):
    filtered = report
    if multihead:
        filtered = report[report["Encoder"].str.contains("Multi")]

    encoders = set(filtered["Encoder"].unique()) - set([""])

    # score_ranks = compute_rank(
    #     filtered, target="Encoder", metric="Score", direction="max"
    # )
    # opt_time_ranks = compute_rank(
    #     filtered, target="Encoder", metric="Opt Train Time Total", direction="min"
    # )
    # def_time_ranks = compute_rank(
    #     filtered, target="Encoder", metric="Default Train Time", direction="min"
    # )

    tune_time = dict(
        zip(
            enc_stats["Model"],
            enc_stats["Total Tune Runtime"].argsort(),
        )
    )
    enc_params = dict(
        zip(
            enc_stats["Model"],
            enc_stats["Encoder Params"].argsort(),
        )
    )
    recon_acc = dict(
        zip(
            enc_stats["Model"],
            enc_stats["Test Recon Accuracy"].argsort()[::-1],
        )
    )

    clf_acc = {}
    if multihead:
        clf_acc = dict(
            zip(
                enc_stats["Model"],
                enc_stats["Test Predict Accuracy"].argsort()[::-1],
            )
        )

    ranks = pd.DataFrame(
        [
            {"Criteria": "Tune Time", **{enc: tune_time[enc] for enc in encoders}},
            {
                "Criteria": "Encoder Params",
                **{enc: enc_params[enc] for enc in encoders},
            },
            {"Criteria": "Recon Accuracy", **{enc: recon_acc[enc] for enc in encoders}},
        ]
        + (
            [{"Criteria": "FT Accuracy", **{enc: clf_acc[enc] for enc in encoders}}]
            if multihead
            else []
        )
        # + [
        #     {
        #         "Criteria": "Clf Score",
        #         **{
        #             enc: score_ranks[score_ranks["Encoder"] == enc]["Rank"].mean()
        #             for enc in encoders
        #         },
        #     },
        #     {
        #         "Criteria": "Clf Opt Train Time",
        #         **{
        #             enc: opt_time_ranks[opt_time_ranks["Encoder"] == enc]["Rank"].mean()
        #             for enc in encoders
        #         },
        #     },
        #     {
        #         "Criteria": "Clf Def. Train Time",
        #         **{
        #             enc: def_time_ranks[def_time_ranks["Encoder"] == enc]["Rank"].mean()
        #             for enc in encoders
        #         },
        #     },
        # ]
    )

    ranks.iloc[:, 1:] = len(encoders) - ranks.iloc[:, 1:]

    return ranks


def enc_compare_plot(report: pd.DataFrame, enc_stats: pd.DataFrame):
    base_ranks = get_enc_ranks(report, enc_stats, False)
    multi_ranks = get_enc_ranks(report, enc_stats, True)

    radar_factory(len(base_ranks))
    fig, axes = plt.subplots(
        nrows=1, ncols=2, subplot_kw=dict(projection="radar"), figsize=(8, 3.5)
    )
    fig.subplots_adjust(wspace=0.25, hspace=0.20, top=0.85, bottom=0.05)

    print("Base Ranks")
    print(base_ranks)
    print()

    make_radarplot(
        data=base_ranks.values[:, 1:].T,
        variables=base_ranks.values[:, 0],
        ax=axes[0],
    )

    print("Multi Ranks")
    print(multi_ranks)
    print()

    make_radarplot(
        data=multi_ranks.values[:, 1:].T,
        variables=multi_ranks.values[:, 0],
        ax=axes[1],
    )

    legend = axes[0, 0].legend(
        set(base_ranks.columns) - set(["Criteria"]),
        loc=(0.9, 0.95),
        labelspacing=0.1,
        fontsize="small",
    )
    fig.savefig(FIGURES_DIR / "enc_compare_radar.pdf", bbox_inches="tight")


@hydra.main(version_base=None, config_path=RUN_CONFIG_DIR, config_name="all")
def run(config):
    report, incomplete = load_all_clf_perf(config)
    enc_stats = load_all_enc_stats(config)
    enc_compare_plot(report, enc_stats)


if __name__ == "__main__":
    run()
