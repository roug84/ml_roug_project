import os
import subprocess

import requests
from roug_ml.utl.paths_utl import create_dir



def download_kaggle_dataset(dataset: str, username: str, key: str, download_path: str) -> None:
    """
    Function to download a dataset from Kaggle competitions.

    :param dataset: The name of the Kaggle competition dataset to download.
    :param username: The Kaggle username of the user.
    :param key: The Kaggle API key for the user.
    :param download_path: The path to the directory where the dataset will be downloaded.
    """
    # Set Kaggle credentials
    os.environ['KAGGLE_USERNAME'] = username
    os.environ['KAGGLE_KEY'] = key

    # Download the dataset
    command = f'kaggle competitions download -c {dataset} -p {download_path}'
    subprocess.call(command, shell=True)


def download_file(in_url: str, file_path: str) -> None:
    """
    Downloads a file from the given URL and saves it to the specified file path.

    :param in_url: The URL of the file to be downloaded.
    :param file_path: The path where the downloaded file should be saved.

    :returns: None
    """
    create_dir(os.path.dirname(file_path))
    response = requests.get(in_url, stream=True)
    response.raise_for_status()

    with open(file_path, 'wb') as file:
        for chunk in response.iter_content(chunk_size=8192):
            file.write(chunk)

    print(f"File downloaded from {in_url} to {file_path}")

    # Example usage:
    # download_file("https://example.com/path/to/file.zip", "/path/to/your/directory/file.zip")
