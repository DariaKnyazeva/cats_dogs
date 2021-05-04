from abc import ABC, abstractmethod
import joblib
import os
from collections import defaultdict

import dask.array as da
import numpy as np
from skimage.io import imread
from sklearn.model_selection import train_test_split
from skimage.transform import resize


class BaseImageFeatureExtractor(ABC):
    """
    Class for extracting features from image data.
    """

    @abstractmethod
    def transform_image_to_dataset(self, image_paths, image_size=(150, 150), batch_size=1000):
        """
        Rescales image from the given filepath to the provided size
        and transforms it to an array.
        @param image_paths: list of full paths to images
        @param image_size: size for image scaling
        @param batch_size: batch size to split large dataset into
        Returns generator of (X, y)
        - **X** : array of image data
        - **y**: array of labels
        """
        pass

    @abstractmethod
    def combine_batches(self, batches, verbose=True):
        """
        Combines large dataset batches into train, validation datasets.
        If verbose is True prints stats on the dataset.
        """
        pass


class ImageFeatureExtractor(BaseImageFeatureExtractor):
    """
    Class for extracting features from image data.
    """

    def _batch_slicer(self, iterable, size=1000):
        batch = []

        for i, item in enumerate(iterable, start=1):
            batch.append(item)
            if i % size == 0:
                yield batch
                batch = []

        if batch:
            yield batch

    def _filter_per_batch(self, batch, image_size=(150, 150)):
        img_data = defaultdict(list)
        for filepath in batch:
            label = self._get_label_from_filepath(filepath)

            im = imread(filepath)
            im = resize(im, image_size)
            img_data['label'].append(label)
            img_data['data'].append(im)
        X = np.array(img_data['data'])
        y = np.array(img_data['label'])

        return X, y

    def transform_image_to_dataset(self, image_paths,
                                   image_size=(150, 150),
                                   batch_size=1000):
        for batch in self._batch_slicer(image_paths, int(batch_size)):
            X, y = self._filter_per_batch(batch, image_size=image_size)
            yield (X, y)

    def _get_label_from_filepath(self, filepath):
        return os.path.split(filepath)[-1][:3]

    def combine_batches(self, batches, verbose=True):
        Xy = list(zip(*batches))
        X_large = da.concatenate([X for X in Xy[0]], axis=0)
        joblib.dump(X_large, "xlarge.pkl")
        y_large = da.concatenate([y for y in Xy[1]], axis=0)
        joblib.dump(y_large, "ylarge.pkl")
        if verbose:
            print(f"X shape {X_large.shape}")
            print(f"y shape {y_large.shape}")
        X_train, X_test, y_train, y_test = train_test_split(
            X_large,
            y_large,
            test_size=0.2,
            shuffle=True,
            random_state=42,
        )
        if verbose:
            print(f"Train dataset shape {X_train.shape}")
            print(f"Test dataset shape {X_test.shape}")
            print(f"Labels {da.unique(y_large).compute()}")

        return (X_train, y_train, X_test, y_test)
