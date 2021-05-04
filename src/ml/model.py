"""
Module for prediction models
"""
from abc import ABC, abstractmethod
from enum import Enum
import numpy as np
from typing import List, Optional, TypeVar

import joblib
from dask_ml.wrappers import ParallelPostFit
from sklearn import svm
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report

from src.ml.sklearn.image_transformers import HogTransformer, RGB2GrayTransformer


ModelClassifier = TypeVar('ModelClassifier')


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
    def load(self, source: str) -> ModelClassifier:
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


class SKLinearImageModel(BaseModel):
    """
    Image classification model that uses sklearn.linear_modeSGDClassifier.
    """

    def __init__(self,
                 pkl_file: Optional[str],
                 probability_threshold: float = 0.75) -> None:
        """
        - **probability_threshhold** : Prediction probability threshold to predict the class label
        - **pkl_file** : Pickle file to load trained model from. If None,
                         use sklearn.linear_model.SGDClassifier classifier.
        """
        self.probability_threshold = probability_threshold

        self.grayify = RGB2GrayTransformer()
        self.hogify = HogTransformer(
            None,
            pixels_per_cell=(8, 8),
            cells_per_block=(3, 3),
            orientations=9,
            block_norm='L2-Hys'
        )
        self.scalify = StandardScaler()

        if pkl_file is None:
            self.classifier = SGDClassifier(random_state=42,
                                            max_iter=1500,
                                            tol=1e-3,
                                            loss='log')
        else:
            self.classifier = self.load(pkl_file)

    def save(self, destination: str) -> None:
        joblib.dump(self.classifier, destination)

    def load(self, source: str) -> ModelClassifier:
        try:
            classifier = joblib.load(source)
            if classifier is None:
                raise ModelException("Unable to load classifier")
            return classifier
        except FileNotFoundError:
            raise ModelException(f"File {source} not found")

    def _preprocess_dataset(self, X: np.ndarray, need_scale: bool = True) -> np.ndarray:
        """
        Transform dataset from RGB to gray, then to hog and scale it if need_scale.
        Returns transformed array.
        """
        X_gray = self.grayify.fit_transform(X)
        X_hog = self.hogify.fit_transform(X_gray)
        if need_scale:
            return self.scalify.fit_transform(X_hog)
        else:
            return X_hog

    def train(self,
              X_train: np.ndarray,
              y_train: np.ndarray,
              X_test: np.ndarray,
              y_test: np.ndarray,
              verbose: bool = True,
              optimize: bool = False):
        X_train_prepared = self._preprocess_dataset(X_train)

        clf = ParallelPostFit(self.classifier, scoring='accuracy')
        self.classifier = clf.fit(X_train_prepared, y_train)

        X_test_prepared = self._preprocess_dataset(X_test)
        prediction = self.classifier.predict_proba(X_test_prepared)

        y_proba_list = [self._predict_proba_to_label(proba).value for proba in prediction]

        if verbose:
            self.evaluate(y_proba_list, y_test, classes=self.classifier.classes_)

        if optimize:
            opt_classifier = ImageModelOptimiser(self).optimize(X_train, y_train)
            prediction = opt_classifier.predict(X_test_prepared)
            y_proba_list = [self._predict_proba_to_label(proba) for proba in prediction]
            if verbose:
                self.evaluate(prediction, y_test)
            self.classifier = opt_classifier

    def _predict_proba_to_label(self, proba: List[float]) -> LABEL:
        if proba[0] >= self.probability_threshold:
            return LABEL.CAT
        elif proba[1] >= self.probability_threshold:
            return LABEL.DOG
        else:
            return LABEL.UNKNOWN

    def predict(self, X: np.ndarray) -> List[LABEL]:
        X_prepared = self._preprocess_dataset(X, need_scale=False)
        prediction = self.classifier.predict_proba(X_prepared)
        y_proba_list = [self._predict_proba_to_label(proba) for proba in prediction]
        return y_proba_list

    def evaluate(self, prediction: np.ndarray,
                 y_test: np.ndarray, classes: List[str]) -> None:
        """
        Prints out model accuracy report.
        """
        print(f'Accuracy: {accuracy_score(y_test, prediction):.2f}')
        print(classification_report(y_test, prediction,
                                    labels=classes))


class ImageModelOptimiser:
    """
    Image classifier model optimization
    """

    def __init__(self, model: BaseModel) -> None:
        self.HOG_pipeline = Pipeline([
            ('grayify', model.grayify),
            ('hogify', model.hogify),
            ('scalify', model.scalify),
            ('classify', model.classifier)
        ])

        self.param_grid = [
            {
                'hogify__orientations': [8, 9],
                'hogify__cells_per_block': [(2, 2), (3, 3)],
                'hogify__pixels_per_cell': [(8, 8), (10, 10), (12, 12)]
            },
            {
                'hogify__orientations': [8],
                'hogify__cells_per_block': [(3, 3)],
                'hogify__pixels_per_cell': [(8, 8)],
                'classify': [
                    model.classifier,
                    svm.SVC(kernel='linear')
                ]
            }
        ]

    def optimize(self, X: np.ndarray, y: np.ndarray) -> ModelClassifier:
        """
        Run fit with all sets of parameters.
        Returns the most optimal classifier.
        """
        grid_search = GridSearchCV(self.HOG_pipeline,
                                   self.param_grid,
                                   cv=3,  # cross validation: 3 folds
                                   n_jobs=-1,
                                   scoring='accuracy',
                                   verbose=1,
                                   return_train_score=True)

        return grid_search.fit(X, y)
