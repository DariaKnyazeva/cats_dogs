from typing import List, Optional, Type, Tuple

import numpy as np
import tensorflow as tf

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    tf.config.experimental.set_memory_growth(
        gpus[0], True)
else:
    print("Can't find GPU. TF will be work on CPU")

from tensorflow.keras.models import load_model, save_model
from tensorflow.keras.applications import EfficientNetB3
from tensorflow.keras.models import Model

from .model import BaseModel, LABEL


class TFModel(BaseModel):

    def __init__(self, pre_train: Optional[str] = None,
                 _class: Type[Model] = EfficientNetB3,
                 input_shape: Tuple[int, int, int] = (224, 224, 3),
                 probability_threshold=0.8):
        self.image_shape = input_shape
        self.threshold = probability_threshold
        if pre_train:
            self.classifier = self.load(pre_train)
        else:
            model = EfficientNetB3(input_shape=input_shape, weights=None, classes=2)
            optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
            model.compile(optimizer=optimizer,
                          loss='sparse_categorical_crossentropy',
                          metrics=['acc'])

    def load(self, source: str) -> Model:
        """load tf model"""
        self.classifier = load_model(source)
        return self.classifier

    @property
    def input_shape(self):
        return self.classifier.input_shape[1:]

    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_test: np.ndarray, y_test: np.ndarray,
              verbose: bool = True) -> None:
        raise NotImplemented

    def _labeling(self, data: np.array) -> List[LABEL]:
        result = []
        for y1, y2 in data:
            if y1 >= self.threshold:
                result.append(LABEL.DOG)
            elif y2 >= self.threshold:
                result.append(LABEL.CAT)
            else:
                result.append(LABEL.UNKNOWN)
        return result

    def predict(self, X: np.ndarray) -> List[LABEL]:
        predictions = self.classifier.predict(X)
        predictions = self._labeling(predictions)
        return predictions

    def save(self, destination: str) -> None:
        """save tf model"""
        save_model(self.classifier, destination)
