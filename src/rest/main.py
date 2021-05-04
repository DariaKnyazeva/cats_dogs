import numpy as np

from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel
from PIL import Image, ImageOps
from skimage.transform import resize

from src.ml.label import LABEL
from src.ml.sklearn_model import SKLinearImageModel

APP = FastAPI()

IMAGE_SIZE = (150, 150)
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
    label: LABEL


@APP.post("/predict/sklinear", summary="Predict the image class with SKLinearImageModel",
          description=DESCRIPTION, response_model=DataResponse)
async def predict_sklinear(image: UploadFile = File(...), ):
    """
    Predicts image class: 'cat', 'dog', 'unknown', or 'unsupported'.
    Use SKLinearImageModel for prediction
    - **image** : *.jpg file
    """
    if image.filename[-3:] not in ('jpg', 'png'):
        return DataResponse(**{"filename": image.filename, 'label': LABEL.UNSUPPORTED})

    img = Image.open(image.file)
    im = ImageOps.exif_transpose(img)
    im = np.array(im)

    im = resize(im, IMAGE_SIZE)
    X = np.array([im, ])
    prediction = SKLINEAR_MODEL.predict(X)
    label = [p.value for p in prediction][0] if prediction else LABEL.UNSUPPORTED

    return DataResponse(**{"filename": image.filename, 'label': label})
