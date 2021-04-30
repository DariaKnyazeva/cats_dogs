import os
import shutil

from fastapi import FastAPI, File, UploadFile

from file_parsers import DirectoryParser
from ml.sklearn import ImageFeatureExtractor
from ml import SKLinearImageModel

app = FastAPI()

SKLINEAR_MODEL = SKLinearImageModel(load_model=True)

USER_UPLOADS = 'user_uploads'

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

    if not os.path.exists(USER_UPLOADS):
        os.mkdir(USER_UPLOADS)
    filepath = os.path.join(USER_UPLOADS, image.filename)
    with open(filepath, "wb") as buffer:
        shutil.copyfileobj(image.file, buffer)

    dir_parser = DirectoryParser(USER_UPLOADS)
    feature_extractor = ImageFeatureExtractor()
    X, _y = feature_extractor.transform_image_to_dataset(dir_parser.full_path_image_files)
    label = SKLINEAR_MODEL.predict(X)

    return {"filename": image.filename, 'label': label}
