from typing import Literal
import torch
import numpy as np
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from umap import UMAP

def reduce_tsne(
    X_train: torch.Tensor | np.ndarray, 
    X_val: torch.Tensor | np.ndarray | None,
    X_test: torch.Tensor | np.ndarray | None,
    random_seed: int = 0,
) -> dict[str, np.ndarray | None]:
    # if len(X_train_reduce)
    # reducer = TSNE(n_components=2, random_state = random_seed)
    if len(X_train) > 100:
        perplexity = 30    
    else:
        perplexity = len(X_train) /3
    perplexity = max(5, perplexity)

    X_train_reduced = TSNE(
        n_components=2, 
        random_state = random_seed,
        perplexity=perplexity
    ).fit_transform(X_train)
    X_val_reduced = (
        TSNE(
            n_components=2, 
            random_state = random_seed,
            perplexity=perplexity
        ).fit_transform(X_val)
        if X_val is not None else
        None
    )
    X_test_reduced = (
        TSNE(
            n_components=2, 
            random_state = random_seed,
            perplexity=perplexity
        ).fit_transform(X_test)
        if X_test is not None else
        None
    )

    return {
        'train': X_train_reduced,
        'val': X_val_reduced,
        'test': X_test_reduced,
    }

def reduce_umap(
    X_train: torch.Tensor | np.ndarray, 
    X_val: torch.Tensor | np.ndarray | None,
    X_test: torch.Tensor | np.ndarray | None,
    random_seed: int = 0,
) -> dict[str, np.ndarray | None]:
    reducer = UMAP(n_components=2, random_state = random_seed)
    X_train_reduced = reducer.fit_transform(X_train)
    X_val_reduced = (
        reducer.transform(X_val)
        if X_val is not None else
        None
    )
    X_test_reduced = (
        reducer.transform(X_test)
        if X_test is not None else
        None
    )
    
    return {
        'train': X_train_reduced,
        'val': X_val_reduced,
        'test': X_test_reduced,
    }
 
def reduce_pca(
    X_train: torch.Tensor | np.ndarray, 
    X_val: torch.Tensor | np.ndarray | None,
    X_test: torch.Tensor | np.ndarray | None,
    random_seed: int = 0,
) -> dict[str, np.ndarray | None]:
    reducer = PCA(n_components=2, random_state = random_seed)
    X_train_reduced = reducer.fit_transform(X_train)
    X_val_reduced = (
        reducer.transform(X_val)
        if X_val is not None else
        None
    )
    X_test_reduced = (
        reducer.transform(X_test)
        if X_test is not None else
        None
    )
    
    return {
        'train': X_train_reduced,
        'val': X_val_reduced,
        'test': X_test_reduced,
    }

def reduce_2d(
    X_train: torch.Tensor | np.ndarray, 
    reduce_method: Literal['tsne','umap','pca'],
    X_val: torch.Tensor | np.ndarray | None = None,
    X_test: torch.Tensor | np.ndarray | None = None,
    random_seed: int = 0,
) -> dict[str, np.ndarray | None]:

    if reduce_method.lower() == 'tsne':
        reduce_func = reduce_tsne
    elif reduce_method.lower() == 'umap':
        reduce_func = reduce_umap
    elif reduce_method.lower() == 'pca':
        reduce_func = reduce_pca
    else:
        raise NotImplementedError

    return reduce_func(X_train, X_val, X_test, random_seed)
