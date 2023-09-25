from sklearn.metrics import confusion_matrix, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from roug_ml.utl.etl.transforms_utl import one_hot_to_numeric


def compute_multiclass_confusion_matrix(targets: np.ndarray or list,
                                        outputs_list: list,
                                        model_names: list,
                                        class_labels: dict = None):
    """
    Function to compute and plot the confusion matrix for multi-class classification.

    :param targets: Ground truth labels (multi-class)
    :param outputs_list: List of model outputs (one-hot or softmax) for different models
    :param model_names: List of names for each model in outputs_list
    :param class_labels: Dictionary of class labels for confusion matrix tick labels

    :return: None
    """

    if len(outputs_list) != len(model_names):
        raise ValueError("The number of outputs should match the number of model names.")

    # Convert targets if necessary
    if targets.ndim > 1:
        targets = one_hot_to_numeric(targets)

    outputs_list_new = []
    # Check if all targets and outputs have the same shape
    for outputs in outputs_list:
        if outputs.ndim > 1:
            outputs = one_hot_to_numeric(outputs)

        if targets.shape != outputs.shape:
            raise ValueError("The shapes of targets and outputs are not the same.")
        outputs_list_new.append(outputs)

    # If class_labels is provided, extract labels from the dictionary
    tick_labels = None
    if class_labels is not None:
        tick_labels = [class_labels[i] for i in sorted(class_labels.keys())]

    fig, axes = plt.subplots(nrows=1, ncols=len(outputs_list_new),
                             figsize=(8 * len(outputs_list_new), 8))

    if len(outputs_list_new) == 1:
        axes = [axes]

    for idx, output in enumerate(outputs_list_new):
        cm = confusion_matrix(targets, output)

        if tick_labels is not None:
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=tick_labels,
                        yticklabels=tick_labels if idx == 0 else [], ax=axes[idx])
        else:
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx])
            if idx != 0:
                axes[idx].set_yticks([])

        axes[idx].set_title(f"Predicted: {model_names[idx]}")

    plt.tight_layout()
    return fig


def compute_multiclass_f1_score(targets: np.ndarray or list,
                                outputs: np.ndarray or list,
                                average: str = 'micro') -> None:
    """
    Function to compute the F1-score for multi-class classification.

    :param targets: Ground truth labels (multi-class)
    :param outputs: Model outputs (one-hot or softmax)
    :param average: Method to compute F1-score ('micro', 'macro', 'weighted')
    """
    predicted = one_hot_to_numeric(outputs)
    f1 = f1_score(targets, predicted, average=average)

    print(f"F1-score ({average}): {f1}")