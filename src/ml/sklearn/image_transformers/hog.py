from typing import Tuple

import numpy as np
from skimage.feature import hog
from sklearn.base import BaseEstimator, TransformerMixin


class HogTransformer(BaseEstimator, TransformerMixin):
    """
    Expects an array of 2d arrays (1 channel images)
    Calculates hog features for each img
    """

    def __init__(self, y: np.ndarray = None,
                 orientations: int = 9,
                 pixels_per_cell: Tuple[int, int] = (8, 8),
                 cells_per_block: Tuple[int, int] = (3, 3),
                 block_norm: str = 'L2-Hys') -> None:
        self.y = y
        self.orientations = orientations
        self.pixels_per_cell = pixels_per_cell
        self.cells_per_block = cells_per_block
        self.block_norm = block_norm

    def fit(self, X: np.ndarray, y: np.ndarray = None) -> TransformerMixin:
        return self

    def transform(self, X: np.ndarray, y: np.ndarray = None) -> np.ndarray:

        def local_hog(X):
            return hog(X,
                       orientations=self.orientations,
                       pixels_per_cell=self.pixels_per_cell,
                       cells_per_block=self.cells_per_block,
                       block_norm=self.block_norm)
        return np.array([local_hog(img) for img in X])
