"""
Scrip for training model and saving it to Pickle
"""

import argparse
import sys
import textwrap

from src.ml import ImageFeatureExtractor, SKLinearImageModel
from src.directory_parser import DirectoryParser


DOC = """
Additional information:
    Path to the directory can be either full, or relative to the script.
    Supported files format: *.jpg, *.png
Example usage:
    python train_model.py --dir data/train --pkl_file 1.pkl
Example output:
    Found 25000 jpg/png file(s) out of 25000 file(s).
    Unsupported files: 0
    X shape (25000, 150, 150, 3)
    y shape (25000,)
    Train dataset shape (20000, 150, 150, 3)
    Accuracy: 0.67
              precision    recall  f1-score   support

         cat       0.67      0.67      0.67      2469
         dog       0.68      0.66      0.67      2531

   micro avg       0.68      0.67      0.67      5000
   macro avg       0.68      0.67      0.67      5000
weighted avg       0.68      0.67      0.67      5000
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
    arg_parser.add_argument('--batch_size', default=1000,
                            help='Batch size for large amount of training data')
    parsed_args = arg_parser.parse_args(sys.argv[1:])

    input_directory = parsed_args.dir
    pkl_file = parsed_args.pkl_file
    batch_size = parsed_args.batch_size

    try:
        dir_parser = DirectoryParser(input_directory)
    except ValueError as e:
        print(*e.args)
        sys.exit(1)

    print(dir_parser)

    model = SKLinearImageModel(pkl_file=None)

    feature_extractor = ImageFeatureExtractor()

    batches = feature_extractor.transform_image_to_dataset(dir_parser.full_path_image_files,
                                                           batch_size=batch_size)
    X_train, y_train, X_test, y_test = feature_extractor.combine_batches(batches=batches)

    model.train(X_train, y_train, X_test, y_test)
    model.save(pkl_file)
