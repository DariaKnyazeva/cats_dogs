import argparse
import sys
import textwrap

from ml.sklearn import ImageFeatureExtractor
from ml import SKLinearImageModel
from file_parsers import DirectoryParser


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
        python cats_and_dogs.py --dir data/demo --pkl_file trained_models/f_12.pkl
"""

if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser(description='Cats and docs recogniser.',
                                         prog='cats_and_dogs',
                                         formatter_class=argparse.RawDescriptionHelpFormatter,
                                         epilog=textwrap.dedent(DOC))
    arg_parser.add_argument('--dir',
                            help='Path to the input directory with image files.')
    arg_parser.add_argument('--pkl_file',
                            help='file name to load the trained model from')
    parsed_args = arg_parser.parse_args(sys.argv[1:])

    input_directory = parsed_args.dir
    pkl_file = parsed_args.pkl_file

    try:
        dir_parser = DirectoryParser(input_directory)
    except ValueError as e:
        print(*e.args)
        sys.exit(1)

    print(dir_parser)

    model = SKLinearImageModel(pkl_file=pkl_file, load_model=True)
    feature_extractor = ImageFeatureExtractor()

    for filepath in dir_parser.full_path_image_files:
        X, y = feature_extractor.transform_image_to_dataset([filepath])
        prediction = model.predict(X)
        print(f"{filepath}: {prediction}")
