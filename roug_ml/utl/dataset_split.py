from typing import Tuple
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.data import random_split
from torch.utils.data import Dataset, Subset
from sklearn.model_selection import train_test_split

from typing import Tuple
from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms
from torch.utils.data import random_split, Dataset


def load_transform_image_data_from_path(main_path: str, transform: transforms) -> Dataset:
    """
    Load image data from the given path and apply transforms.

    :param main_path: The path to the main directory containing all the images.
    :param transform: Image transformations to be applied.

    :return: Loaded dataset.
    """
    return ImageFolder(main_path, transform=transform)


def split_image_data(data: Dataset,
                     train_ratio: float,
                     val_ratio: float,
                     ) -> Tuple[Dataset, Dataset]:
    """
    Split data into training, validation, and testing sets.

    :param data: The loaded dataset.
    :param train_ratio: The ratio of data to be used for training.
    :param val_ratio: The ratio of data to be used for validation.
    :return: Training dataset, validation dataset, and testing dataset.
    """

    # Ensure ratios sum to 1
    assert train_ratio + val_ratio == 1, "Ratios must sum to 1"

    # Set the split sizes
    train_size = int(train_ratio * len(data))
    val_size = len(data) - train_size   # remaining for testing

    # Split the data
    train_data, val_data = random_split(data, [train_size, val_size])

    return train_data, val_data


# def split_image_data_from_path(main_path: str,
#                                train_ratio: float,
#                                val_ratio: float,
#                                test_ratio: float,
#                                transform: transforms) -> Tuple[Dataset, Dataset, Dataset]:
#     """
#     Split data into training, validation, and testing sets.
#
#     :param main_path: The path to the main directory containing all the images.
#     :param train_ratio: The ratio of data to be used for training.
#     :param val_ratio: The ratio of data to be used for validation.
#     :param test_ratio: The ratio of data to be used for testing.
#     :param transform:
#
#     :return: Training dataset, validation dataset, and testing dataset.
#     """
#
#     # Load all data
#     all_data = ImageFolder(main_path, transform=transform)
#
#     # Ensure ratios sum to 1
#     assert train_ratio + val_ratio + test_ratio == 1, "Ratios must sum to 1"
#
#     # Set the split sizes
#     train_size = int(train_ratio * len(all_data))
#     val_size = int(val_ratio * len(all_data))
#     test_size = len(all_data) - train_size - val_size  # remaining for testing
#
#     # Split the data
#     train_data, val_data, test_data = random_split(all_data, [train_size, val_size, test_size])
#
#     return train_data, val_data, test_data


def stratified_split(
        dataset: Dataset,
        labels: list,
        test_size: float = 0.2,
        random_state: int = 42) -> Tuple[Subset, Subset]:
    """
    Split a PyTorch Dataset into training and validation datasets using stratified sampling.

    Args:
    - dataset (Dataset): The dataset to split.
    - labels (list): List of labels corresponding to the items in the dataset.
    - test_size (float, optional): Proportion of the dataset to include in the validation split.
      Defaults to 0.2.
    - random_state (int, optional): Random state for reproducibility. Defaults to 42.

    Returns:
    - Tuple[Subset, Subset]: Training and validation datasets.
    """

    # Split indices for stratified sampling
    train_indices, val_indices = train_test_split(
        list(range(len(dataset))),
        test_size=test_size,
        stratify=labels,
        random_state=random_state
    )

    train_subset = Subset(dataset, train_indices)
    val_subset = Subset(dataset, val_indices)

    return train_subset, val_subset

    # How to use:
    # train_dataset, val_dataset = stratified_split(dataset, merged_df['Numeric_Labels'].values)
