import joblib
from typing import List

import numpy as np
from dask_ml.wrappers import ParallelPostFit
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

from ml.model import BaseModel, ModelException, LABEL, T
from ml.sklearn.image_transformers import HogTransformer, RGB2GrayTransformer
from ml.sklearn.model_optimisation import ImageModelOptimiser


class SKLinearImageModel(BaseModel):
    """
    Image classification model that uses sklearn.linear_modeSGDClassifier.
    """

    def __init__(self,
                 probability_threshold: float = 0.75,
                 pkl_file: str = 'trained_models/hog_sklearn.pkl',
                 load_model: bool = True) -> None:
        """
        - **probability_threshhold** : Prediction probability threshold to predict the class label
        - **pkl_file** : Pickle file to load trained model from.
        - **load_model** : If True, loads classifier from the pfovided Pickle,
            Use sklearn.linear_model.SGDClassifier classifier otherwise.
        """
        self.probability_threshold = probability_threshold

        self.grayify = RGB2GrayTransformer()
        self.hogify = HogTransformer(
            pixels_per_cell=(8, 8),
            cells_per_block=(3, 3),
            orientations=9,
            block_norm='L2-Hys'
        )
        self.scalify = StandardScaler()

        if load_model:
            self.classifier = self.load(pkl_file)
        else:
            self.classifier = SGDClassifier(random_state=42,
                                            max_iter=1500,
                                            tol=1e-3,
                                            loss='log')

    def save(self, destination: str) -> None:
        joblib.dump(self.classifier, destination)

    def load(self, source: str) -> T:
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

        y_proba_list = [self._predict_proba_to_label(proba) for proba in prediction]

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
