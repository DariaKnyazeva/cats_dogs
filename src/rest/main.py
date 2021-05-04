import numpy as np

from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel
from skimage.io import imread
from skimage.transform import resize

from src.ml.model import LABEL, SKLinearImageModel

app = FastAPI(title="Cats and Dogs")

SKLINEAR_MODEL = SKLinearImageModel(pkl_file='trained_models/hog_sklearn.pkl')

DESCRIPTION = """
request:\n
    {"image": *.jpg file}
response:\n
    {
       "filename": image filename,
       "label": predicted class label, 'cat', 'dog', 'unknown', or 'unsupported'
    }
"""


class DataResponse(BaseModel):
    """
    - filename - name of file
    - label - class label
    """
    filename: str
    label: str


@app.post("/predict/sklinear",
          summary="Predict the image class with SKLinearImageModel",
          description=DESCRIPTION)
async def predict_sklinear(image: UploadFile = File(...)):
    """
    Predicts image class: 'cat', 'dog', 'unknown', or 'unsupported'.
    Use SKLinearImageModel for prediction
    - **image** : *.jpg file
    """
    if image.filename[-3:] not in ('jpg', 'png'):
        return DataResponse(**{"filename": image.filename, 'label': 'unsupported'})

    im = imread(image.file)
    im = resize(im, (150, 150))
    X = np.array([im, ])
    prediction = SKLINEAR_MODEL.predict(X)
    label = [p.value for p in prediction][0] if prediction else 'unsupported'

    return DataResponse(**{"filename": image.filename, 'label': label})
