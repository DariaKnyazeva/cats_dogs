# cats_dogs
ML exercise to classify cats and dogs images.

Implemented and trained model that uses `sklearn.linear_model.SGDClassifier` classifier.
The input image is resized to 150x150 pixels by default, and transformed into `numpy array`. Some data preprocessing takes place before model training and prediction: RGB is transgormed to grey and then [hogified](https://learnopencv.com/histogram-of-oriented-gradients/).

Started to integrate TensorFlow model class for training; use ready trained model file cats_and_dogs_86.tf. It can be used by specifying in command line script for prediction.

It is possible to use more precise models by inheriting from the abstraction `ml.model.BaseModel`, which provides interface for training and prediction.

## Environment setup:
Use Python 3.7

Install virtual environment:
```
virtualenv -p python3.7 venv
```

Activate virtual environment:
```
source venv/bin/activate
```

Install required python packages:
```
pip install - r requirements.txt
```

## Command line scripts:

### Train the model

To train the model run `train_model.py`. This script uses skitlearn model.

Cats and docs model trainer.

required argument:
  * --dir DIR    Path to the input directory with image files to train the model on.

optional arguments:
  * -h, --help           show this help message and exit
  * --pkl_file PKL_FILE  Pickle file name to save the trained model. The default value is 'trained_models/fitted_model.pkl'
  * --batch_size' BATCH_SIZE  Batch size for large amount of training data. Default is 1000.

additional information:
    Path to the directory can be either full,
                   or relative to the script.
    Supported files format: *.jpg, *.png


Example usage:
`python train_model.py --dir data/train --pkl_file fitted.pkl`

Example output:
```
    Found 414 jpg/png file(s) out of 414 file(s).
    Unsupported files: 0
    number_of_samples: 414
    image_shape: (150, 150, 3)
    labels: ['cat' 'dog']
    labels_distribution: {'cat': 253, 'dog': 161}

    Accuracy: 0.67
              precision    recall  f1-score   support

         cat       0.67      0.67      0.67      2469
         dog       0.68      0.66      0.67      2531

   micro avg       0.68      0.67      0.67      5000
   macro avg       0.68      0.67      0.67      5000
weighted avg       0.68      0.67      0.67      5000
 ```

### Classify images from the provided directory

usage: 
`python cats_and_dogs.py [-h] [-f PKL_FILE] [-p PROBABILITY_THRESHOLD] [-m {ModelUsageType.tf,ModelUsageType.sk}] dir`

Cats and docs recogniser.

optional arguments:
  * -h, --help           show this help message and exit
  * --dir DIR            Path to the input directory with image files.
  * --pkl_file PKL_FILE  Pickle or file name to load the trained model from. Default is 'trained_models/hog_sklearn.pkl'
  * -p, --probability_threshold PROBABILITY_THRESHOLD. Limit to accept probability to predict a class. Default is 0.75
  * -m', '--model_type' MODEL_TYPE. Scikitlearn or Tensorflow model. Default is tf. 

```
additional information:
    Path to the directory can be either full, or relative to the script.
    Supported files format: *.jpg, *.png
    Example output:
        Found 24 jpg/png file(s) out of 25 file(s).
        Unsupported files: {'some_file.txt'}
        image_1.jpg: ['cat']
        image_2.png: ['dog']
        image_3.jpg: ['unknown_class']
        image_4.jpg: ['cat']
    Example usage:
        python cats_and_dogs.py data/demo -m sk -f trained_models/hog_sklearn.pkl
        python cats_and_dogs.py data/demo -m tf
```

## Run unit tests

```
pytest
```

## REST API

### Start web server

`cd cats_dogs`

`source venv/bin/activate`

`uvicorn asgi:app --reload`

### Swagger documentation

Open your browser and see REST API documentation at `http://127.0.0.1:8000/docs`
