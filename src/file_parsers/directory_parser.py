import os


class DirectoryParser:
    """
    Class for walking in the provided directory
    and for checking if it contains images
    """

    def __init__(self, path_to_directory, allowed_extensions=('jpg', 'png',)):
        if not os.path.exists(path_to_directory):
            raise ValueError(f"{path_to_directory} not found.")
        if not os.path.isdir(path_to_directory):
            raise ValueError("f{path_to_directory} is not a directory.")

        self.path_to_directory = path_to_directory
        self.extensions = allowed_extensions

        _dirpath, _dirnames, self.filenames = next(os.walk(self.path_to_directory))

        self.img_filenames = [f for f in self.filenames if f[-3:] in allowed_extensions]
        self.img_paths = [os.path.join(_dirpath, f) for f in self.img_filenames]
        self.not_img_paths = (set(self.filenames) - set(self.img_filenames)) or 0

    def __str__(self):
        if len(self.img_filenames) == 0:
            return f"Did not find any {'/'.join(self.extensions)} files in the {self.path_to_directory}"
        return f"Found {len(self.img_filenames)} {'/'.join(self.extensions)} file(s) out of "\
            f"{len(self.filenames)} file(s).\n"\
            f"Unsupported files: {self.not_img_paths}"

    @property
    def image_files(self):
        return self.img_filenames

    @property
    def full_path_image_files(self):
        return self.img_paths
