import os
from fastapi.testclient import TestClient

from rest.main import app

client = TestClient(app)


def test_post_image():
    base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    fpath = os.path.join(base_path, 'data/dir_test/cat.0.jpg')
    with open(fpath, "rb") as f:
        response = client.post("/predict/sklinear",
                               files={"image": ("filename", f, "image/jpeg")})
        print(response.json())
        assert response.status_code == 200
