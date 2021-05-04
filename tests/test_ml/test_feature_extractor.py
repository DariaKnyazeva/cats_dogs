import os
import unittest
from PIL import Image


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

    def test_rotate_if_transposed(self):
        imgpath = os.path.join(self.base_path, "data/dog_rotated_90_3.jpg")
        testable = ImageFeatureExtractor()
        # import piexif
        # picture = imgpath
        # exif_dict = piexif.load(picture)
        # if piexif.ImageIFD.Orientation in exif_dict["0th"]:
        #     orientation = exif_dict["0th"][piexif.ImageIFD.Orientation]
        #     print(orientation)
        img = Image.open(imgpath)
        print(img.info)
        print(img.width)
        img1 = testable.rotate_if_transposed(imgpath)
        print(img1.width)
        img1.save(os.path.join(self.base_path, "data/dog_rotated.jpg"))
