import os

from file_parsers import DirectoryParser
from tests.async_test_case import AsyncTestCase


class ImageParserTest(AsyncTestCase):
    @classmethod
    def setUpClass(cls):
        cls.base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

    def _test_resize(self):
        path = os.path.join(self.base_path, "data/dir_test")
        dir_parser = DirectoryParser(path)
        test_image = dir_parser.full_path_image_files[0]
        testable = ImageParser()

        test_width = 200
        test_height = 250

        result = self.get_async_result(testable.resize, test_image,
                                       width=test_width, height=test_height)

        keys = ['data', 'filename', 'height', 'label', 'width']
        self.assertListEqual(keys, sorted(list(result.keys())))
        self.assertEqual('cat', result['label'])
        self.assertEqual(test_height, result['height'])
        self.assertEqual(test_width, result['width'])
        self.assertEqual((test_width, test_height, 3), result['data'].shape)

# unittest.main()
