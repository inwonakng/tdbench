from pathlib import Path
import pandas as pd
import shutil
import os
import json
import numpy as np

# Set up kaggle credential
CUR_DIR = Path(__file__).parent
with open(CUR_DIR/'../../../kaggle.json') as f:
    for k,v in json.load(f).items():
        os.environ[f'KAGGLE_{k.upper()}'] = v

from kaggle.api.kaggle_api_extended import KaggleApi

api = KaggleApi()
api.authenticate()

def load_kaggle_dataset(
    dataset_dir: str | Path,
    dataset_name: str,
    file_name: str,
    label: str,
) -> tuple[pd.DataFrame, pd.Series]:

    """Downloades and parses the dataset using Kaggle API. The API key must be stored under `kaggle.json` of the project root.

    Args:
        dataset_dir (str | Path): Directory to save the dataset files.
        dataset_name (str): Name of the dataset. Must be in kaggle format {uploader}/{dataset}
        file_name (str): Name of the file to pull from the dataset repo.
        label (str): Name of the label column in the file.

    Raises:
        NotImplementedError: Thrown if the file type is unknown.

    Returns:
        tuple[pd.DataFrame, np.ndarray]: X in pandas dataframe and y in numpy array.
    """
    dataset_dir = Path(dataset_dir)
    
    download_dir = dataset_dir / 'original'
    source_file = download_dir / Path(file_name).name
    if not source_file.is_file():
        print('downloading file from kaggle...')
        api.dataset_download_file(
            dataset=dataset_name,
            path = dataset_dir / 'original',
            file_name = file_name
        )

    for file in download_dir.glob('*'):
        if file.suffix == '.zip':
            if not Path(file.parent / file.stem).is_file():
                shutil.unpack_archive(file, download_dir)
    
    if source_file.suffix == '.csv':
        df = pd.read_csv(source_file)
    elif source_file.suffix == '.txt':
        df = pd.read_table(source_file)
    else: 
        raise NotImplementedError

    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].astype('category').cat.codes

    df = df[~df[label].isna()].fillna(0)
    X = df.loc[:, df.columns!=label]
    y = df[label].values

    # feature_categ_mask = (X.dtypes == np.int8).values
    # split_mask = get_split_mask(y)

    return X, y
