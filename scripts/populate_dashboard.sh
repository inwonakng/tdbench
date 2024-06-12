source scripts/activate_conda.sh

python -m tdbench.dashboard.dataset_stats_plot
python -m tdbench.dashboard.dd_perf_plot
python -m tdbench.dashboard.enc_perf_plot
python -m tdbench.dashboard.clf_perf_plot
python -m tdbench.dashboard.clf_perf_over_n_plot

cp -r dashboard_assets/* ,,/tdbench.github.io/static
