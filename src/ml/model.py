"""
Module for prediction models
"""
from abc import ABC, abstractmethod
from enum import Enum
import numpy as np
from typing import List, TypeVar


T = TypeVar('T')


class LABEL(str, Enum):
    CAT = 'cat'
    DOG = 'dog'
    UNKNOWN = 'unknown'


class ModelException(Exception):
    pass


class BaseModel(ABC):
    """
    Abstract class for classification model.
    """

    @abstractmethod
    def save(self, destination: str) -> None:
        """
        Saves the fitted classifier to the provided destination
        @param destination: src Pickle file name, or database table name.
        """
        pass

    @abstractmethod
    def load(self, source: str) -> T:
        """
        Loads the fitted classifier from some source, e. g. Pickle
        for the further use on prediction.
        """
        pass

    @abstractmethod
    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_test: np.ndarray, y_test: np.ndarray, verbose: bool = True) -> None:
        """
        Trains the model on the provided dataset.
        """
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> List[LABEL]:
        """
        Returns prediction array for the provided dataset.
        """
        pass
