from pathlib import Path
import pandas as pd
import arff

from .utils import download

ORDINAL_TYPES = ["NUMERIC", "REAL", ""]


def load_openml_dataset(
    dataset_dir: str | Path,
    download_url: str,
    label: str,
    raw_data: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, pd.Series]:
    """Downloads and parses the dataset from openml.

    Args:
        dataset_dir (str | Path): Directory to save the dataset files.
        download_url (str): Link to the openml repository of that dataset.
        label (str): Name of the label column.

    Returns:
        tuple[pd.DataFrame, np.ndarray]: X in pandas dataframe and y in numpy array.
    """

    dataset_dir = Path(dataset_dir)
    dataset_dir.mkdir(parents=True, exist_ok=True)

    download_dir = dataset_dir / "original.arff"
    if not download_dir.is_file():
        download(download_url, download_dir)
    with open(download_dir) as f:
        arff_data = arff.load(f)

    col_names, col_attrs = zip(*arff_data["attributes"])
    df = pd.DataFrame(arff_data["data"], columns=col_names)

    if raw_data:
        return df[~df[label].isna()].loc[:, df.columns != label], df[label]

    # remove exclude columns and also where the target col is nan
    df = df[~df[label].isna()].fillna(0)

    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].astype("category").cat.codes

    # split into label col and the rest
    X = df.loc[:, df.columns != label]
    y = df[label]
    return X, y
