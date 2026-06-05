import numpy as np
from scipy.stats import rankdata


class RRFBasedFusion():

    def __init__(self, smoothing_param: int=60):
        self._smoothing_param = smoothing_param

    
    def fuse(self, feature_dists: list) -> np.ndarray:
        final_dists = np.sum(np.array(feature_dists), axis=0)
        print(final_dists)

        return final_dists
    

    def normalize_distances(self, dists: np.ndarray) -> np.ndarray:
        return np.apply_along_axis(lambda row: rankdata(row) + self._smoothing_param, axis=1, arr=dists)