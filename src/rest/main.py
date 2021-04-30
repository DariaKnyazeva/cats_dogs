import os
import shutil

from fastapi import FastAPI, File, UploadFile

from file_parsers import DirectoryParser
from ml.sklearn import ImageFeatureExtractor
from ml import SKLinearImageModel

app = FastAPI()

SKLINEAR_MODEL = SKLinearImageModel(load_model=True)


@app.post("/predict/sklinear")
async def predict_sklinear(image: UploadFile = File(...),
                           summary="Predict the image class with SKLinearImageModel"):
    """
    Predicts image class: 'cat', 'dog', 'uncnown_class', or 'unsupported'.
    Use SKLinearImageModel for prediction
    - **image** : *.jpg file
    """
    # TODO: process file in memory without storying to disc
    filepath = os.path.join("user_uploads", image.filename)
    with open(filepath, "wb") as buffer:
        shutil.copyfileobj(image.file, buffer)
    label = 'unknown_class'

    try:
        dir_parser = DirectoryParser("user_uploads")
    except ValueError:
        return {"filename": image.filename, 'label': 'unsupported'}

    for filepath in dir_parser.full_path_image_files:
        feature_extractor = ImageFeatureExtractor([filepath])
        label = SKLINEAR_MODEL.predict(feature_extractor.X)

    return {"filename": image.filename, 'label': label}
