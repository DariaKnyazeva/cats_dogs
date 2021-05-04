from abc import ABC, abstractmethod
from collections import defaultdict
import os

import dask.array as da
import joblib
import numpy as np
from PIL import Image, ImageOps
from typing import Generator, Iterable, Tuple, List
from skimage.io import imread
from sklearn.model_selection import train_test_split
from skimage.transform import resize


class BaseImageFeatureExtractor(ABC):
    """
    Class for extracting features from image data.
    """

    @abstractmethod
    def transform_image_to_dataset(self,
                                   image_paths: Iterable[str],
                                   image_size: Tuple[int, int] = (150, 150),
                                   batch_size: int = 1000) -> Generator[Tuple[np.ndarray, np.ndarray],
                                                                        None,
                                                                        None]:
        """
        Rescales image from the given filepath to the provided image size
        and transforms it to an array. Yields batches of image arrays.
        Labels are extracted from image paths (filename starts with a label).

        @param image_paths: iterable of full paths to images
        @param image_size: size for image scaling
        @param batch_size: batch size to split large dataset into
        Returns generator of (X, y)
        - **X** : array of image data
        - **y**: array of labels
        """
        pass

    @abstractmethod
    def combine_batches(self, batches: Iterable[np.ndarray],
                        verbose: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Combines large dataset batches into train, validation datasets.
        If verbose is True prints stats on the dataset.
        """
        pass


class ImageFeatureExtractor(BaseImageFeatureExtractor):
    """
    Class for extracting features from image data.
    """

    def _batch_slicer(self,
                      iterable: Iterable,
                      size: int = 1000) -> Generator[list, None, None]:
        batch = []

        for i, item in enumerate(iterable, start=1):
            batch.append(item)
            if i % size == 0:
                yield batch
                batch = []

        if batch:
            yield batch

    # def _transform_image_per_batch(self,
    #                                batch: Iterable[str],
    #                                image_size: Tuple[int, int] = (150, 150)) -> Tuple[np.ndarray,
    #                                                                                   np.ndarray]:
    def _transform_image_per_batch(self, batch, image_size=(150, 150)):
        img_data = defaultdict(list)
        for filepath in batch:
            label = self._get_label_from_filepath(filepath)

            im = imread(filepath)
            im = resize(im, image_size)
            correct_size = image_size + (3, )
            if im.shape != correct_size:
                print(f"Corrupted image {filepath}")
                continue
            img_data['label'].append(label)
            img_data['data'].append(im)
        y = np.array(img_data['label'])
        X = np.array(img_data['data'])
        return X, y

    def transform_image_to_dataset(self,
                                   image_paths: Iterable[str],
                                   image_size: Tuple[int, int] = (150, 150),
                                   batch_size: int = 1000) -> List[Tuple[np.ndarray, np.ndarray]]:
        result = []
        for batch in self._batch_slicer(image_paths, int(batch_size)):
            X, y = self._transform_image_per_batch(batch, image_size=image_size)
            result.append((X, y))
        return result

    def _get_label_from_filepath(self, filepath: str) -> str:
        """
        Gets class label from filepath.
        File name starts with the label in training data sets.
        """
        return os.path.split(filepath)[-1][:3]

    def combine_batches(self, batches: Iterable[np.ndarray],
                        verbose: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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

    def rotate_if_transposed(self, imgpath: str) -> Image:
        """
        If imgpath does not exist, returns None.

        If an image has an EXIF Orientation tag, return a new image
        that is transposed accordingly. Otherwise, return a copy of the image.
        """
        if not os.path.exists(imgpath):
            return None

        img = Image.open(imgpath)
        return ImageOps.exif_transpose(img)
