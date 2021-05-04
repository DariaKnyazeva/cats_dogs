import numpy as np

from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel
from PIL import Image, ImageOps
from skimage.transform import resize

from ml import SKLinearImageModel
from ml.model import LABEL

app = FastAPI(title="Cats and Dogs")

SKLINEAR_MODEL = SKLinearImageModel(load_model=True)

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


@app.post("/predict/sklinear")
async def predict_sklinear(image: UploadFile = File(...),
                           summary="Predict the image class with SKLinearImageModel",
                           description=DESCRIPTION):
    """
    Predicts image class: 'cat', 'dog', 'unknown', or 'unsupported'.
    Use SKLinearImageModel for prediction
    - **image** : *.jpg file
    """
    if image.filename[-3:] not in ('jpg', 'png'):
        return {"filename": image.filename, 'label': 'unsupported'}

    img = Image.open(image.file)
    im = ImageOps.exif_transpose(img)
    im = np.array(im)

    im = resize(im, (150, 150))
    X = np.array([im, ])
    prediction = SKLINEAR_MODEL.predict(X)
    label = prediction[0] if prediction else 'unsupported'

    return DataResponse(**{"filename": image.filename, 'label': label})
