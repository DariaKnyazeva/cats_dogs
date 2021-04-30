from abc import ABC, abstractmethod


class BaseImageFeatureExtractor(ABC):
    """
    Class for extracting features from image data.
    """

    @abstractmethod
    def transform_image_to_dataset(self, image_paths, image_size=(150, 150)):
        """
        Rescales image from the given filepath to the provided size
        and transforms it to an array.
        @param image_paths: list of full paths to images
        Returns arrays X, y
        - **X** : array of image data
        - **y**: array of labels
        """
        pass

    @abstractmethod
    def get_stats(self):
        """
        Spits out summary statistics on the image data.
        Returns dict.
        """
        pass

    @abstractmethod
    def print_stats(self):
        """
        Prints the summary statistics
        """
        pass
