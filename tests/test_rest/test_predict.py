import os
import unittest
from fastapi.testclient import TestClient

from src.rest.main import app

client = TestClient(app)


class RestApiTest(unittest.TestCase):
    def test_predict_sklinear_unsupported(self):
        base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        fpath = os.path.join(base_path, 'data/dir_test/1.txt')
        with open(fpath, "rb") as f:
            response = client.post("/predict/sklinear",
                                   files={"image": ("1.txt", f, "image/jpeg")})
            self.assertEqual(200, response.status_code)
            content = response.json()
            self.assertEqual('1.txt', content['filename'])
            self.assertEqual('unsupported', content['label'])

    def test_predict_sklinear(self):
        base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        fpath = os.path.join(base_path, 'data/dir_test/cat.780.jpg')
        with open(fpath, "rb") as f:
            response = client.post("/predict/sklinear",
                                   files={"image": ("cat.780.jpg", f, "image/jpeg")})
            self.assertEqual(200, response.status_code)
            content = response.json()
            self.assertEqual('cat.780.jpg', content['filename'])
            self.assertEqual('cat', content['label'])
