import os
import zipfile
import glob
import shutil
from typing import List, Union


def extract_zip(file_path: Union[os.PathLike, str],
                extraction_path: Union[os.PathLike, str]) -> None:
    """
    This function extracts a zip file.

    :param file_path: str or os.PathLike, the path of the zip file.
    :param extraction_path: str or os.PathLike, the path where the zip file should be extracted.
    """
    with zipfile.ZipFile(file_path, 'r') as zip_ref:
        zip_ref.extractall(extraction_path)
    print(f"File extracted to {extraction_path}")

    # Example usage:
    # extract_zip(os.path.join(res_path, 'dogs-vs-cats.zip'), res_path)


def move_files_to_labels_dir(original_path: Union[os.PathLike, str], labels: List[str]) -> None:
    """
    This function moves image files to their corresponding folders based on the labels.

    :param original_path: str or os.PathLike, the path where the image files are located.
    :param labels: list of str, the labels corresponding to the files.
    """
    def create_dir(path: Union[os.PathLike, str]) -> None:
        if not os.path.exists(path):
            os.makedirs(path)

    # Create directories based on labels
    for label in labels:
        create_dir(os.path.join(original_path, label))

    # Get all jpg files in the original_path
    files = glob.glob(os.path.join(original_path, '*.jpg'))

    # Move each file to its corresponding directory
    for filename in files:
        basename = os.path.basename(filename)  # Get the base file name
        label = basename.split('.')[0]  # Assume the label is the first part before '.'
        if label in labels:
            new_path = os.path.join(original_path, label)  # Create new path: "original_path/label"
            shutil.move(filename, os.path.join(new_path, basename))  # Moves the file

    print(f"Files moved to their corresponding directories in {original_path}")

    # Example usage:
    # move_files_to_labels_dir(os.path.join(res_path, 'train'), ['cat', 'dog', 'bird'])
