"""
Scrip for training model and saving it to Pickle
"""

import argparse
import sys
from sklearn.model_selection import train_test_split
import textwrap

from ml.sklearn import ImageFeatureExtractor, SKLinearImageModel
from file_parsers import DirectoryParser


DOC = """
Additional information:
    Path to the directory can be either full, or relative to the script.
    Supported files format: *.jpg, *.png
Example usage:
    python train_model.py --dir data/train --pkl_file 1.pkl
Example output:
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
"""

if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser(description='Cats and docs model trainer.',
                                         prog='Image recognition model trainer',
                                         formatter_class=argparse.RawDescriptionHelpFormatter,
                                         epilog=textwrap.dedent(DOC))
    arg_parser.add_argument('--dir',
                            help='Path to the input directory with image files to train the model on.')
    arg_parser.add_argument('--pkl_file', default='trained_models/fitted_model.pkl',
                            help='Pickle file name to save the trained model')
    parsed_args = arg_parser.parse_args(sys.argv[1:])

    input_directory = parsed_args.dir
    pkl_file = parsed_args.pkl_file
    try:
        dir_parser = DirectoryParser(input_directory)
    except ValueError as e:
        print(*e.args)
        sys.exit(1)

    print(dir_parser)

    model = SKLinearImageModel(pkl_file=pkl_file, load_model=False)

    feature_extractor = ImageFeatureExtractor()
    X, y = feature_extractor.transform_image_to_dataset(dir_parser.full_path_image_files)
    feature_extractor.print_stats()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        shuffle=True,
        random_state=42,
    )

    model.train(X_train, y_train, X_test, y_test)
    model.save(pkl_file)
