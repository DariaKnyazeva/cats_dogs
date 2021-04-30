# cats_and_dogs
ML exercise to classify cats and dogs images.

Implemented and trained model that uses `sklearn.linear_model.SGDClassifier` classifier.
The trained model does not have decent accuracy due to low laptop capacity process large amount of data.
It is possible to use more precise models by inheriting from the abstraction `ml.model.BaseModel`.

## Environment setup:
Use Python 3.7

Add src to python path (Linux):

`export PYTHONPATH=$PYTHONPATH:src`

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

    Accuracy: 0.58
              precision    recall  f1-score   support

          cat       0.83      0.60      0.70        58
          dog       0.43      0.52      0.47        25
unknown_class       0.00      0.00      0.00         0

     accuracy                           0.58        83
    macro avg       0.42      0.37      0.39        83
 weighted avg       0.71      0.58      0.63        83
 ```

### Classify images from the provided directory

usage: 
`python cats_and_dogs [-h] [--dir DIR] [--pkl_file PKL_FILE]`

Cats and docs recogniser.

optional arguments:
  * -h, --help           show this help message and exit
  * --dir DIR            Path to the input directory with image files.
  * --model_file MODEL_FILE  Pickle or tf file name to load the trained model from
  * --model {sklearn,tensorflow}
                        Model class to use for prediction. The default is sklearn

```
additional information:
    Path to the directory can be either full, or relative to the script.
    Supported files format: *.jpg, *.png
    Example output:
        Found 24 jpg/png file(s) out of 25 file(s).
        Unsupported files: {'some_file.txt'}
        image_1.jpg: cat
        image_2.png: dog
        image_3.jpg: unknown_class
        image_4.jpg: cat
    Example usage:
        python cats_and_dogs.py --dir data/demo --pkl_file trained_models/f_12.pkl
```

## Run unit tests

```
pytest
```

### REST API

## Start web server

cd to cats_and_dogs directory, activate virtual environemnt and run 

`uvicorn rest.main:app --reload`

Open your browser and see REST API documentation at `http://127.0.0.1:8000/docs`
