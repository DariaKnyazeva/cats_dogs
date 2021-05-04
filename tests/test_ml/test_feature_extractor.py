import os
import unittest


from ml import ImageFeatureExtractor


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
        batches = testable.transform_image_to_dataset(self.image_paths)
        for (X, y) in batches:
            self.assertTupleEqual((2, ) + self.dimension, X.shape)
            self.assertTupleEqual((2, ), y.shape)
