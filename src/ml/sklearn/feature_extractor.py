import os
import numpy as np
from skimage.io import imread
from skimage.transform import resize
from collections import Counter, defaultdict

from ml.feature_extractor import BaseImageFeatureExtractor


class ImageFeatureExtractor(BaseImageFeatureExtractor):
    """
    Class for extracting features from image data.
    """

    def transform_image_to_dataset(self, image_paths, image_size=(150, 150)):
        self.image_data = defaultdict(list)

        for filepath in image_paths:
            label = self._get_label_from_filepath(filepath)

            im = imread(filepath)
            im = resize(im, image_size)
            self.image_data['label'].append(label)
            self.image_data['data'].append(im)

        X = np.array(self.image_data['data'])
        y = np.array(self.image_data['label'])

        return X, y

    def _get_label_from_filepath(self, filepath):
        return os.path.split(filepath)[-1][:3]

    @property
    def features(self):
        return self.X

    @property
    def labels(self):
        return self.y

    def get_stats(self):
        """
        Spits out summary statistics on the image data.
        Returns dict.
        """
        result = dict()

        labels = np.unique(self.image_data['label'])
        img_data = self.image_data['data'][0].shape if self.image_data['data'] else ()

        result['number_of_samples'] = len(self.image_data['data'])
        result['image_shape'] = img_data
        result['labels'] = labels
        result['labels_distribution'] = {k: v for k, v in Counter(self.image_data['label']).items()}
        return result

    def print_stats(self):
        stats = self.get_stats()
        for key, value in stats.items():
            print(f"{key}: {value}")
