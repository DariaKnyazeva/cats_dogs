import os
import unittest


from ml.sklearn import ImageFeatureExtractor


class TestImageFeatureExtractor(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        img_path_1 = os.path.join(cls.base_path, "data/dir_test/cat.780.jpg")
        img_path_2 = os.path.join(cls.base_path, "data/dir_test/dog.2582.jpg")
        cls.image_paths = [img_path_1, img_path_2]
        cls.dimension = (150, 150, 3)

    def test_get_transform_image_to_dataset(self):
        testable = ImageFeatureExtractor()
        X, y = testable.transform_image_to_dataset(self.image_paths)
        self.assertTupleEqual((2, ) + self.dimension, X.shape)
        self.assertTupleEqual((2, ), y.shape)

    def test_get_stats(self):
        testable = ImageFeatureExtractor()
        testable.transform_image_to_dataset(self.image_paths)
        stats = testable.get_stats()
        self.assertEqual(2, stats['number_of_samples'])
        self.assertTupleEqual(self.dimension, stats['image_shape'])
        self.assertListEqual(['cat', 'dog'], list(stats['labels']))
        self.assertDictEqual({'cat': 1, 'dog': 1},
                             stats['labels_distribution'])
