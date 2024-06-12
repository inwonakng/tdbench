import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.neighbors import NearestCentroid


def get_closest_point(
    clustering: AgglomerativeClustering,
    points: np.ndarray,
    point_idxs: np.ndarray,
) -> np.ndarray:
    centroids = NearestCentroid().fit(points, clustering.labels_)
    return [
        point_idxs[clustering.labels_ == l][
            ((c - points[clustering.labels_ == l]) ** 2).sum(1).argmin()
        ]
        for l, c in enumerate(centroids.centroids_)
    ]


def agglomerative(
    X: np.ndarray,
    y: np.ndarray,
    N: int,
    random_state: int | np.random.Generator = 0,
    match_balance: bool = False,
    get_closest: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    clustered_X, clustered_y, clustered_idxs = [], [], []
    all_idxs = np.arange(len(X))
    for label in np.unique(y):
        # break
        clustering = AgglomerativeClustering(
            n_clusters=N,
            # random_state= random_state,
            # n_init= 'auto'
        ).fit(X[y == label])

        if get_closest:
            idxs = get_closest_point(
                clustering=clustering,
                points=X[y == label],
                point_idxs=all_idxs[y == label],
            )
            clustered_idxs.append(idxs)
            clustered_X.append(X[idxs])
        else:
            centroids = NearestCentroid().fit(X[y == label], clustering.labels_)
            clustered_X.append(centroids.centroids_)
        clustered_y += [label] * N

    if get_closest:
        clustered_idxs = np.hstack(clustered_idxs)
    else:
        clustered_idxs = None

    clustered_X = np.vstack(clustered_X)
    clustered_y = np.array(clustered_y)
    return clustered_X, clustered_y, clustered_idxs
