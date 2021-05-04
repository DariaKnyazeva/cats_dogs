import numpy as np
from skimage.color import rgb2gray
from sklearn.base import BaseEstimator, TransformerMixin


class RGB2GrayTransformer(BaseEstimator, TransformerMixin):
    """
    Convert an array of RGB images to grayscale
    """

    def __init__(self) -> None:
        pass

    def fit(self, X: np.ndarray, y: np.ndarray = None) -> TransformerMixin:
        """returns itself"""
        return self

    def transform(self, X: np.ndarray, y: np.ndarray = None) -> np.ndarray:
        """perform the transformation and return an array"""
        return np.array([rgb2gray(img) for img in X])
