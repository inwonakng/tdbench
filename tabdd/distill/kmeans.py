from typing import Optional, Tuple
import numpy as np
from sklearn.cluster import KMeans


def get_closest_point(
    clustering: KMeans,
    points: np.ndarray,
    point_idxs: np.ndarray,
) -> np.ndarray:
    return [
        point_idxs[clustering.labels_ == l][
            ((c - points[clustering.labels_ == l]) ** 2).sum(1).argmin()
        ]
        for l, c in enumerate(clustering.cluster_centers_)
    ]


def kmeans(
    X: np.ndarray,
    y: np.ndarray,
    N: int,
    random_state: int | np.random.Generator = 0,
    match_balance: bool = False,
    get_closest: bool = False,
) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    clustered_X, clustered_y, clustered_idxs = [], [], []
    all_idxs = np.arange(len(X))
    for label in np.unique(y):
        # break
        clustering = KMeans(n_clusters=N, random_state=random_state, n_init="auto").fit(
            X[y == label]
        )

        if not get_closest:
            clustered_X.append(clustering.cluster_centers_)
        else:
            idxs = get_closest_point(
                clustering=clustering,
                points=X[y == label],
                point_idxs=all_idxs[y == label],
            )
            clustered_idxs.append(idxs)
            clustered_X.append(X[idxs])
        clustered_y += [label] * N

    clustered_X = np.vstack(clustered_X)
    if get_closest:
        clustered_idxs = np.hstack(clustered_idxs)
    else:
        clustered_idxs = None
    clustered_y = np.array(clustered_y)
    return clustered_X, clustered_y, clustered_idxs
