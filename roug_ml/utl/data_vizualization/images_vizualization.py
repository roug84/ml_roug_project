import matplotlib.pyplot as plt
import torch
import numpy as np
from matplotlib import pyplot as plt


def imshow(img: np.ndarray or torch.tensor,
           title: str or None = None):
    """
    Displays an image using matplotlib with a title.

    :param img: image to display, can be a NumPy array or a PyTorch tensor
    :type img: numpy.ndarray or torch.Tensor
    :param title: title of the image, defaults to None
    :type title: str, optional
    """
    # If img is a Tensor, convert it to a NumPy array
    if torch.is_tensor(img):
        img = img.numpy()
        # If img is normalized, denormalize it for visualization:
        # The ImageNet means are [0.485, 0.456, 0.406] for RGB, and the standard deviations are
        # [0.229, 0.224, 0.225].
        img = img * np.array([0.229, 0.224, 0.225]).reshape((3, 1, 1)) + np.array(
            [0.485, 0.456, 0.406]).reshape((3, 1, 1))
        img = np.clip(img, 0, 1)
        # PyTorch tensors assume the color channel is the first dimension
        # but matplotlib assumes is the third dimension
        img = np.transpose(img, (1, 2, 0))
    plt.imshow(img)
    if title is not None:
        plt.title(title)
    plt.show()


def visualize_incorrect_classifications(model: torch.nn.Module,
                                        x_test: np.ndarray or torch.Tensor,
                                        y_test: np.ndarray or torch.Tensor) -> None:
    """
    Visualizes the first 10 incorrect classifications made by the model on the test set.

    :param model: Trained model used for making predictions.
    :param x_test: Test set images.
    :param y_test: True labels for the test set images.
    """
    # Ensure y_test is numpy array for comparison
    if isinstance(y_test, torch.Tensor):
        y_test = y_test.numpy()

    predictions = model.predict(x_test)

    # Ensure predictions is numpy array for comparison
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.numpy()

    incorrect_indices = np.nonzero(predictions != y_test)[0]

    plt.figure(figsize=(10, 5))
    for i, incorrect_index in enumerate(incorrect_indices[:10]):
        ax = plt.subplot(2, 5, i + 1)
        img = x_test[incorrect_index]
        # If img is a Tensor, convert it to a NumPy array
        if torch.is_tensor(img):
            img = img.numpy()
        # Change (C, H, W) to (H, W, C)
        img = np.transpose(img, (1, 2, 0))
        ax.imshow(img, cmap='gray')
        ax.set_title(
            f"Predicted: {predictions[incorrect_index]}, Actual: {y_test[incorrect_index]}")
        plt.tight_layout()
    plt.show()
