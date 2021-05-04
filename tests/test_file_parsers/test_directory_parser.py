import os
import unittest

from src.directory_parser import DirectoryParser


class DirectoryParserTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

    def test_path_not_exist(self):
        with self.assertRaises(ValueError):
            DirectoryParser("nosuchpath")

    def test_path_is_not_dir(self):
        file_path = os.path.join(self.base_path, "data/dir_test/1.txt")
        with self.assertRaises(ValueError):
            DirectoryParser(file_path)

    def test_dir_empty(self):
        path = os.path.join(self.base_path, "data/dir_test_empty")
        if not os.path.exists(path):
            os.mkdir(path)
        testable = DirectoryParser(path)
        self.assertEqual([], testable.img_filenames)
        msg = f"Did not find any jpg/png files in the {path}"
        self.assertEqual(msg, str(testable))

    def test_dir_no_images(self):
        path = os.path.join(self.base_path, "data/dir_test_no_images")
        testable = DirectoryParser(path)
        self.assertEqual([], testable.img_filenames)
        msg = f"Did not find any jpg/png files in the {path}"
        self.assertEqual(msg, str(testable))

    def test_dir_with_images(self):
        path = os.path.join(self.base_path, "data/dir_test")
        testable = DirectoryParser(path)
        self.assertEqual(['cat.780.jpg', 'dog.2582.jpg'], testable.img_filenames)
        msg = f"Found 2 jpg/png file(s) out of 3 file(s).\n"\
            "Unsupported files: {'1.txt'}"
        self.assertEqual(msg, str(testable))


# unittest.main()
