import numpy as np
def random_sample(
    X: np.ndarray,
    y: np.ndarray,
    N: int | float, 
    random_state: int | np.random.Generator = 0,
    match_balance: bool = False,
):
    np.random.seed(random_state)
    sampled_X, sampled_y = [],[]
    y_unique, y_counts  = np.unique(y, return_counts=True)

    if N > 1:
        if match_balance:
            sample_sizes = (y_counts / y_counts.sum() * N).astype(int)
        else:
            sample_sizes = [int(N)] * len(y_unique)
    else:
        if match_balance:
            sample_sizes = (y_counts * N).astype(int)
        else:
            sample_sizes = [int(len(y) * N)] * len(y_unique)

    for label, ss in zip(y_unique, sample_sizes):
        sampled_X.append(
            X[y == label][
                np.random.choice(
                    np.arange(
                        (y==label).sum()
                    ), 
                    ss, 
                    replace=False
                )
            ]
        )

        sampled_y += [label] * ss

    sampled_X = np.vstack(sampled_X)
    sampled_y = np.array(sampled_y)

    return sampled_X, sampled_y
