"""
Module for prediction models
"""
from abc import ABC, abstractmethod
from enum import Enum


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
    def save(self, destination):
        """
        Saves the fitted classifier to the provided destination
        @param destination: src Pickle file name, or database table name.
        """
        pass

    @abstractmethod
    def load(self, source):
        """
        Loads the fitted classifier from some source, e. g. Pickle
        for the further use on prediction.
        """
        pass

    @abstractmethod
    def train(self, X_train, y_train, X_test, y_test, verbose=True):
        """
        Trains the model on the provided dataset.
        """
        pass

    @abstractmethod
    def predict(self, X):
        """
        Returns prediction array for the provided dataset.
        """
        pass
