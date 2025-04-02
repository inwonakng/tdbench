from sklearn.base import BaseEstimator
from xgboost import XGBClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from tabpfn import TabPFNClassifier

from .ft_transformer import ScikitFTTransformer
from .resnet import ScikitResNet


def get_classifier(
    classifier_name: str,
    default_params: dict,
    params: dict,
) -> BaseEstimator:
    sample_dset = params.pop("sample_dset", None)
    model_params = {**default_params, **params}
    if classifier_name.lower() == "xgbclassifier":
        return XGBClassifier(**model_params)
    elif classifier_name.lower() == "mlpclassifier":
        return MLPClassifier(**model_params)
    elif classifier_name.lower() == "logisticregression":
        return LogisticRegression(**model_params)
    elif classifier_name.lower() == "kneighborsclassifier":
        return KNeighborsClassifier(**model_params)
    elif classifier_name.lower() == "gaussiannb":
        return GaussianNB(**model_params)
    elif classifier_name.lower() == "tabpfn":
        return TabPFNClassifier(**model_params)
    elif classifier_name.lower() == "fttransformer":
        if sample_dset is None:
            raise ValueError("sample_dset is required for FTTransformer")
        return ScikitFTTransformer(**model_params, sample_dset=sample_dset)
    elif classifier_name.lower() == "resnet":
        if sample_dset is None:
            raise ValueError("sample_dset is required for Resnet")
        return ScikitResNet(**model_params, sample_dset=sample_dset)
    else:
        raise ValueError(f"Unknown classifier name: {classifier_name}")
