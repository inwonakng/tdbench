# TDBench: Tabular Data Distillation Benchmark

It contains all the code and resources necessary to recreate the experiments described in the paper titled **TDBench: Tabular Data Distillation Benchmark**.

## Installation

Conda version: 23.3.1
Python version: 3.10

You can create a python 3.10 environment named `tdbench` by running:

```bash
conda create -n tdbench python=3.10
```

It can be activated by running

```bash
conda activate tdbench
```

Only torch and torch_geometric need to be installed through conda, the rest works through pip. (torch and torch_geometric may also work with pip, but the project was only tested with conda installations)

Install torch by activating the environment and running:

```bash
conda install pytorch:pytorch -c pytorch
```

torch_geometric can be installed by running:

```bash
conda install pyg -c pyg
```

Then you can install following packages through pip by running:

```bash
pip install -r requirements_x86.txt
```

If running on a mac, you need to install by `requirements_osx.txt`

## Running the code

The `.env` file in the root directory of the project must be populated with the following variables:

```bash
HDF5_USE_FILE_LOCKING=FALSE
DATA_REPO_DIR={DIR_TO_STORE_DATA}
DASHBOARD_DATA_REPO_DIR={DIR_TO_STORE_DASHBOARD_CHARTS}
RAY_TMP_DIR={TMP_DIR_FOR_RAY}
RAY_RESULT_DIR={RESULT_DIR_FOR_RAY}
KAGGLE_USERNAME={KAGGLE_USERNAME}
KAGGLE_KEY={KAGGLE_API_KEY}
CONDA_ENV={CONDA_ENV_NAME_OR_PATH}
```

Every script in the project must be ran in module mode with the `-m` flag, otherwise the dependencies will break.
The configuration is managed through hydra.

All configuration is managed by hydra, and train/optimizing autoencoders and downstream classifiers can be done with the `scripts/tune.sh` script.
This script can be ran from anywhere, and assumes that conda base environment is activated.

### Arguments and default values

- `--op`
  - Operation to perform.
  - Options: ["classifier", "encoder"].
  - Defaults to "classifier".
- `--datasets`
  - Datasts to use.
  - Options: Any dataset correctly configured under `config/data/datasets`.
  - Defaults to all available datasets.
- `--data_mode`
  - Mode to parse the data in.
  - Options ["onehot", "mixed", "onehot-mixed", "mixed-onehot"].
  - Defaults to "onehot"
- `--classifiers`
  - Classifiers to train. Only relevant when `operation=="classifier"`.
  - Options: Any classifier configured under `config/classifier/models`.
    Defaults to all classifiers.
- `--distill_methods`
  - Distillation methods to use. Only relevant when operation=="classifier".
  - Options: Any distillation method configured under `config/classifier/distill_methods`.
  - Defaults to all distillation methods.
- `--encoders`
  - Encoders to train.
  - Options: Any encoder configured under `config/encoder/models`.
  - Defaults to all encoders.
- `--encoder_train`
  - Encoder training setting.
  - Options: Any setting configured under `config/encoder/train`.
  - Defaults to "default"
- `--encoder_train_target`
  - Encoder training target.
  - Options: ["[base]", "[multihead]", "[base,multihead]"].
  - Defaults to ["base"].
- `--latent_dim`
  - Latent dimension of the encoder.
  - Defaults to 16.
- `--checkpoint_dir`
  - Directory to store autoencoder checkpoints.
  - Defaults to "best_checkpoints".
- `--results_dir`
  - Directory to store results.
  - Defaults to "$DATA_REPO_DIR/tune_classifier_results".
- `--tune_hyperopt`
  - Flag to enable hyperopt optimization for downstream classifiers.
  - If included, will set to "true"

### Training autoencoders

```bash
bash scripts/tune.sh \
  --op=encoder \
  --datasets=[adult] \
  --encoders=[tf] \
  --encoder_train=a100s \
  --encoder_train_target=[multihead] \
  --latent_dim=16 \
  --checkpoint_dir=best_checkpoints;
```

### Training classifiers

```bash
bash scripts/tune.sh \
  --op=classifier \
  --datasets=[adult] \
  --encoders=none \
  --classifiers=[xgb] \
  --distill_methods=[kmeans] \
  --results_dir=reports \
  --tune_hyperopt;
```

## Files

**Data Files**

These files are not included in the repository due to their sizes.

- `data_mode_switch_results_w_reg.csv`
  - Contains the results of every run.
  - Contains information such as the scores, runtime and various parameters of the run.
  - Download link: [https://drive.google.com/file/d/1DPIGMo1_4iwYXMchMZPXBnIujsSfc5I_/view?usp=share_link](https://drive.google.com/file/d/1DPIGMo1_4iwYXMchMZPXBnIujsSfc5I_/view?usp=share_link)
- `ds_stats.csv`
  - Contains dataset statistics.
  - Download link: [https://drive.google.com/file/d/1_0p3gZ47y5gfrwoTDE51eZ_xn3nEC3YH/view?usp=share_link](https://drive.google.com/file/d/1_0p3gZ47y5gfrwoTDE51eZ_xn3nEC3YH/view?usp=share_link)
- `enc_stats.csv`
  - Contains statistics of the encoder models.
  - Download link: [https://drive.google.com/file/d/1u3SsQ9p3OiiOX1AakbnJqQIYx5AfYA7d/view?usp=share_link](https://drive.google.com/file/d/1u3SsQ9p3OiiOX1AakbnJqQIYx5AfYA7d/view?usp=share_link)

These files are required for the analysis code. 
Once these files have been downloaded to the project root, the following command will conduct the analysis as seen in the paper.

```bash
python -m tdbench.scripts.analyze_results
```

**Scripts**

- `analyze_results.py`
  - Driver code that produces most of the results seen in the main paper.
  - Conducts the analysis in the order seen in the manuscript.
- `classifier_performance.py`
  - Helper code to parse/rank the results by different groups.
- `ft_resnet_runtime.py`
  - Script to measure the average runtime of the FTTransformer and ResNet downstream classifiers.

