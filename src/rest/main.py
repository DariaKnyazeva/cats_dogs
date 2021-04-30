import numpy as np

from fastapi import FastAPI, File, UploadFile
from skimage.io import imread
from skimage.transform import resize

from ml import SKLinearImageModel

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

    im = imread(image.file)
    im = resize(im, (150, 150))
    X = np.array([im, ])
    label = SKLINEAR_MODEL.predict(X)

    return {"filename": image.filename, 'label': label}
