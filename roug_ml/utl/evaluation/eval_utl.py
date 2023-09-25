import torch
from sklearn.metrics import accuracy_score
# import tensorflow as tf
from sklearn.metrics import confusion_matrix, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import log_loss, accuracy_score
from roug_ml.utl.etl.transforms_utl import one_hot_to_numeric


def compute_accuracy_from_soft_max(y, y_pred):
    return accuracy_score(y, tf.one_hot(tf.argmax(y_pred, axis=1), depth=13).numpy(), normalize=True)


def compute_confusion_matrix(targets, outputs):
    """
    Function to compute and plot the confusion matrix for binary classification.

    :param targets: Ground truth labels (binary)
    :type targets: numpy.array or list
    :param outputs: Model outputs (logits or probabilities for the positive class)
    :type outputs: numpy.array or list

    :return: None
    """
    predicted = (outputs > 0.5).astype(int)
    cm = confusion_matrix(targets, predicted)

    # Plot confusion matrix
    plt.figure(figsize=(5, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()


def compute_f1_score(targets, outputs):
    """
    Function to compute the F1-score for binary classification.

    :param targets: Ground truth labels (binary)
    :type targets: numpy.array or list
    :param outputs: Model outputs (logits or probabilities for the positive class)
    :type outputs: numpy.array or list

    :return: None
    """
    predicted = (outputs > 0.5).astype(int)
    f1 = f1_score(targets, predicted)

    print(f"F1-score: {f1}")


def calc_loss_acc(outputs, targets, cost_function):
    """
    Calculate loss and accuracy for the given model outputs and targets.
    :param outputs: The outputs produced by the model. Should be an instance of torch.Tensor.
    :param targets: The ground truth targets that the model aims to predict. Should be an instance
    of torch.Tensor.
    :param cost_function: The cost function used to evaluate the model's predictions. Should be a
    callable function.

    :return: A tuple where the first element is the calculated loss value (float) and the second
    element is the calculated accuracy (float).
    """
    if cost_function is not None:
        loss = cost_function(outputs, targets).item()
    else:
        loss = None

    # Check if outputs are probabilities or class labels
    if outputs.dim() == 1 or outputs.size(1) == 1:
        predicted = outputs
    else:
        # Get the index of the max log-probability (which is the predicted class)
        _, predicted = torch.max(outputs, 1)

    if targets.dim() > 1:  # Targets are one-hot encoded
        targets = targets.argmax(dim=1)

    # Compute how many of these predictions were correct by comparing with the targets
    correct = (predicted == targets).sum().item()

    # Accuracy is the number of correct predictions divided by the total number of samples
    acc = correct / targets.size(0)

    return loss, acc


def calc_loss_acc_val(outputs, targets):
    """
    Calculate loss and accuracy for the given model outputs and targets.
    :param outputs: The outputs produced by the model. Should be an instance of numpy.ndarray.
    :param targets: The ground truth targets that the model aims to predict. Should be an instance
    of numpy.ndarray.

    :return: A tuple where the first element is the calculated loss value (float) and the second
    element is the calculated accuracy (float).
    """

    # Convert targets and outputs to predicted class labels if necessary
    if outputs.ndim > 1:
        outputs = one_hot_to_numeric(outputs)
    if targets.ndim > 1:
        targets = one_hot_to_numeric(targets)

    # Check if targets and outputs have the same shape
    if targets.shape != outputs.shape:
        raise ValueError("The shapes of targets and outputs are not the same.")

    acc = accuracy_score(targets, outputs)

    return acc
