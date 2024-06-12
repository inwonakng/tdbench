import numpy as np
def is_onehot(X):return not bool(set(np.unique(X)) - set([0,1]))
