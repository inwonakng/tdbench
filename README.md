# tab-data-distill

Repository for the extern project on Tabular Data Distillation

## Installation

Conda is recommended for this repository

Conda version: 23.3.1
Python version: 3.10

You can create a python 3.10 environment named `tabdd` by running:

```bash
conda create -n tabdd python=3.10
```

It can be activated by running

```bash
conda activate tabdd
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
pip install -r requirements.txt
```

If running on a mac, you need to install by `requirements_mac.txt`

## Running the code

The .env file in the root directory of the project must be populated with the following variables:

```bash
HDF5_USE_FILE_LOCKING=FALSE
DATA_REPO_DIR={DIR_TO_STORE_DATA}
DASHBOARD_DATA_REPO_DIR={DIR_TO_STORE_DASHBOARD_CHARTS}
RAY_TMP_DIR={TMP_DIR_FOR_RAY}
RAY_RESULT_DIR={RESULT_DIR_FOR_RAY}
KAGGLE_USERNAME={KAGGLE_USERNAME}
KAGGLE_KEY={KAGGLE_API_KEY}
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
  - Defaults to "npl"
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
  --encoder_train=npl \
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

### Using Hydra

If you wish to manually use the python script, you can run individual scripts (under `tabdd/scripts/` or `tabdd/tune`) with the module mode as the following:

```bash
python -u -m tabdd.{SCRIPT_NAME} $ARGUMENTS}
```

Note that the `/` in the script path must be changed to `.`.

For example, a downstream classifier testing for XGBClassifier and Logistic Regression with HPO enabled can be ran with the following command:

```bash
python -m tabdd.tune.classifier \
  data/datasets=[mini_boo_ne] \
  data.mode.parse_mode=mixed \
  classifier/models=[xgb,logistic_regression] \
  distill/methods=[original] \
  distill/common=n_100 \
  classifier.train.cpu_per_worker=1 \
  classifier.train.results_dir=hpo_measure \
  classifier.train.tune_hyperopt=true;
```

If you check the entry point of the script (`__main__`), you will find that the configuration is built off of a particular file.
In this case, the `tabdd/tune/classifier.py` script is based on the `config/tune.yaml` file.
To override the entire configuration, you may specify the path using slashes.
e.g.:

    data/datasets=[mini_boo_ne] \

However, if you only want to change a flag in the default configuration, you can specify it by using dots.
e.g.:

    classifier.train.tune_hyperopt=true \

## Extending this project

### Changing default parameters

The configurations for this project are managed by hydra and can be modified by adding new files/directories under the `config` directory.
For example, the example seen above:

```bash
python -m tabdd.tune.classifier \
  data/datasets=[mini_boo_ne] \
  data.mode.parse_mode=mixed \
  classifier/models=[xgb,logistic_regression] \
  distill/methods=[original] \
  distill/common=n_100 \
  classifier.train.cpu_per_worker=1 \
  classifier.train.results_dir=hpo_measure \
  classifier.train.tune_hyperopt=true;
```

overrides the default configuration for `distill/common` with a file named `n_100`.
The default value for `distill/common` is the following:

```yaml
random_iters: 5
distill_sizes:
  - 10
  - 20
  - 30
  - 40
  - 50
  - 60
  - 70
  - 80
  - 90
  - 100
```

However, the `distill/n_100.yaml` can look something like this:

```yaml
random_iters: 1
distill_sizes:
  - 100
```

Similarly, any configuration in the project can be tweaked in this way.

### Adding new datasets

Adding new datasets is as simple as adding a new `config/data/datasets/{DATASET_NAME}.yaml` file.
Currently, only openml datasets are supported.

The following flags must be specified for the dataset to be correctly loaded:

```yaml
DATASET_NAME:
  _target_: tabdd.config.DatasetConfig
  dataset_name: ... # string
  download_url: ... # string
  label: ... # string
  n_classes: ... # int
  source_type: ... # string
```

### Adding new preprocessing methods

The preprocessing is handled by the `TabularDataModule` object that lives in `tabdd/data/tabulardatamodule.py`.

The preprocessing strategies are identified by a string, and can be configured under `config/data/mode`.
An example configuration for the `onehot` setting is as follows:

```yaml
_target_: tabdd.config.DataModeConfig
parse_mode: onehot
scale_mode: standard
bin_strat: uniform
n_bins: 10
batch_size: 1024
val_ratio: 0.15
test_ratio: 0.15
```

This setting uses the `standard` scaler and `uniform` binning strategy with 10 bins.
One can additionally define any type of `scale_mode` or `bin_strat`, which will be consumed by the `TabularDataModule`.

This object is configured with `DatasetConfig` and `DataModeConfig`.
The `DatasetConfig` is the configuration for the dataset, and the `DataModeConfig` is the configuration for the preprocessing method.

It's `TabularDataModule.prepare_data` is the method that will parse the data accordingly and save to cache.
One can add arbitrary preprocessing methods in this file by adding new flags to `DataModeConfig` and handling it inside the `prepare_data` method.

### Adding new distillation methods

The distillation methods are identified by a string, which should have a configuration with the same name under `config/distill/methods`.
Once can characterize the method the following fields:

- `is_random`: Whether there is randomness in the method. If true, the pipeline will be ran multiple times.
- `is_cluster`: Whether the method is a clustering method. If true, an option that uses the nearest-to-center method will be included.
- `can_use_encoder`: Whether the method can be applied in the latent space.
- `args`: any additional arguments to the actual function.

Below is an example configuration for KMeans

```yaml
kmeans:
  _target_: tabdd.config.DistillConfig
  distill_method_name: kmeans
  is_random: true
  is_cluster: true
  is_baseline: false
  can_use_encoder: true
  can_distill: true
  args: {}
```

Once the configuration is created, it will be consumed by `load_distilled_data` method of `tabdd/distill/load_distilled_data.py`.
This method can then be modified to include the new distillation method.

### Adding new encoders

All encoders used in the benchmark are subclasses `BaseEncoder` from `tabdd/models/encoder/base_encoder.py`. A simple example of how to implement can be seen in `tabdd/models/encoder/mlp_autoencoder.py`. The module needs to encoder the following methods: `__init__()`, `encode`, `decode` and `forward`.

The autoencoders are specified by the configuration files in `config/encoder/models/`. An example configuration for the MLP autoencoder is as follows:
```yaml
mlp:
  _target_: tabdd.config.EncoderTuneConfig
  encoder_name: MLPAutoEncoder
  cls:
    _target_: hydra.utils.get_class
    path: tabdd.models.encoder.MLPAutoEncoder
  tune_params:
    encoder_dims:
      _target_: ray.tune.choice
      categories:
        - [100]
        - [100, 100]
        - [100, 100, 100]
        - [100, 100, 100, 100]
        - [200]
        - [200, 200]
        - [200, 200, 200]
        - [200, 200, 200, 200]
    decoder_dims:
      _target_: ray.tune.choice
      categories:
        - [100]
        - [100, 100]
        - [100, 100, 100]
        - [100, 100, 100, 100]
        - [200]
        - [200, 200]
        - [200, 200, 200]
        - [200, 200, 200, 200]
    embed_dim:
      _target_: ray.tune.choice
      categories: [10, 20, 50, 100, 200]
    dropout_p:
      _target_: ray.tune.choice
      categories: [0, 0.2, 0.4]
    use_embedding:
      _target_: ray.tune.choice
      categories: [true, false]
    opt_name:
      _target_: ray.tune.choice
      categories: [Adam]
    opt_lr:
      _target_: ray.tune.choice
      categories: [0.1, 0.01, 0.001]
    opt_wd:
      _target_: ray.tune.choice
      categories: [0, 0.001, 0.0001]
    sch_names:
      _target_: ray.tune.choice
      categories:
        - []
        - [ReduceLROnPlateau]
        - [CosineAnnealingLR]
```

Notice that the class of the encoder is specified by `cls`, and the hyperparameters are specified by `tune_params`.

### Reproducing results seen in `On Learning Representations for Tabular Data Distillation`

**Data repository**

Using the following files in the repository, you may reproduce every table and plot in the main manuscript:

- [dataset_stats.csv](./dataset_stats.csv)
- [enc_stats.csv](./enc_stats.csv)
- [data_mode_switch_results.csv](./data_mode_switch_results.csv)
- [hpo-measure/](./hpo-measure/)
- [mixed_tf_results.csv](./mixed_tf_results.csv)
- [ple_tf_results.csv](./ple_tf_results.csv)

**Script files**

These scripts are loosely organized by the RQs answered in the paper.
These can be simply ran by calling `python {SCRIPT_NAME}`.

- [Q0_experiment_scale.py](./Q0_experiment_scale.py)
- [Q1_1_col_embeds.py](./Q1_1_col_embeds.py)
- [Q1_encoding.py](./Q1_encoding.py)
- [Q2_distill_methods.py](./Q2_distill_methods.py)
- [Q3_autoencoders.py](./Q3_autoencoders.py)
- [Q4_1_runtime.py](./Q4_1_runtime.py)
- [Q4_2_get_hpo_dirs.py](./Q4_2_get_hpo_dirs.py)
- [Q4_2_hpo.py](./Q4_2_hpo.py)
- [Q4_combinations.py](./Q4_combinations.py)
- [Q5_class_imbal.py](./Q5_class_imbal.py)
