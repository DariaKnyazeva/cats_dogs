import argparse
import sys
import textwrap
from enum import Enum

from src.ml import ImageFeatureExtractor, SKLinearImageModel
from src.directory_parser import DirectoryParser
from src.ml.tf_model import TFModel

DOC = """
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
        python cats_and_dogs.py data/demo -m tf
        python cats_and_dogs.py data/demo -m sk -f trained_models/hog_sklearn.tf
"""


class ModelUsageType(Enum):
    tf = "tf"
    sk = 'sk'


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser(description='Cats and docs recogniser.',
                                         prog='cats_and_dogs',
                                         formatter_class=argparse.RawDescriptionHelpFormatter,
                                         epilog=textwrap.dedent(DOC))
    arg_parser.add_argument('-f', '--pkl_file',
                            help='file name to load the trained model from',
                            default='trained_models/cat_dog_model_0.86.tf')
    arg_parser.add_argument('-p', '--probability_threshold', default=0.75,
                            help='Limit to accept probability to predict a class.')
    arg_parser.add_argument('-m', '--model_type', default=ModelUsageType.tf,
                            choices=list(ModelUsageType), type=ModelUsageType,
                            help='Limit to accept probability to predict a class.')
    arg_parser.add_argument('dir',
                            help='Path to the input directory with image files.')
    parsed_args = arg_parser.parse_args(sys.argv[1:])

    input_directory = parsed_args.dir
    pkl_file = parsed_args.pkl_file
    probability_threshold = float(parsed_args.probability_threshold)

    try:
        dir_parser = DirectoryParser(input_directory)
    except ValueError as e:
        print(*e.args)
        sys.exit(1)

    print(f"Data dir: {dir_parser}")

    feature_extractor = ImageFeatureExtractor()
    if parsed_args.model_type == ModelUsageType.sk:
        model = SKLinearImageModel(pkl_file=pkl_file,
                                   probability_threshold=probability_threshold)
        X, _ = feature_extractor.transform_image_to_dataset(dir_parser.full_path_image_files)[0]
    else:
        model = TFModel(pre_train=pkl_file,
                        probability_threshold=probability_threshold)
        X, _ = feature_extractor.transform_image_to_dataset(dir_parser.full_path_image_files,
                                                            image_size=model.image_shape[:-1])[0]
    prediction = model.predict(X)
    for filepath, prediction in zip(dir_parser.full_path_image_files, prediction):
        print(f"{filepath}: {prediction.value}")
