from typing import Literal
from sklearn.preprocessing import KBinsDiscretizer

def build_one_hot_encoder(
    strategy: Literal['uniform', 'quantile', 'kmeans'] | str, 
    n_bins: int
) -> KBinsDiscretizer:
    return KBinsDiscretizer(n_bins = n_bins, strategy=strategy, random_state=0)
