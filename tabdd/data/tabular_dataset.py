from typing import Optional
import numpy as np
import torch
from torch.utils.data import Dataset

class TabularDataset(Dataset):
    X: torch.Tensor
    y: torch.FloatTensor
    feature_mask: np.ndarray
    feature_categ_mask: np.ndarray
    feature_idx: torch.Tensor

    def __init__(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        feature_mask: np.ndarray,
        feature_categ_mask: np.ndarray,
        feature_idx: Optional[torch.Tensor] = None,
    ) -> None:
        self.X = X
        self.y = y
        self.feature_mask = feature_mask
        self.feature_categ_mask = feature_categ_mask
        self.feature_idx = feature_idx

    def __len__(self) -> int:
        return self.y.shape[0]

    def __getitem__(
        self, index: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.feature_idx is None:
            return self.X[index], self.y[index]
        else:
            return self.X[index], self.y[index], self.feature_idx[index]

