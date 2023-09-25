import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from torch.utils.data import Dataset
from beartype.typing import Tuple, Dict, List, Union

import matplotlib.pyplot as plt
from collections import Counter
import numpy as np
from typing import Dict, Tuple


def plot_label_distribution_from_arrays(
        label_mapping: Dict[int, str],
        **data: Tuple[np.ndarray, np.ndarray]
) -> plt.Figure:
    """
    Plots the distribution of labels in the provided data and returns the figure.

    Args:
    - label_mapping (Dict[int, str]): A dictionary mapping from numeric labels to string labels.
    - data (Tuple[np.ndarray, np.ndarray]): Named data to visualize. Should be in the form (X, y).
    """

    barWidth = 0.15
    colors = ['blue', 'red', 'green', 'yellow', 'purple']
    fig, ax = plt.subplots(figsize=(15, 8))

    all_labels = set(int(label) for _, y in data.values() for label in y)
    str_labels = sorted(list(all_labels), key=lambda x: label_mapping[x])
    labels_str = [label_mapping[label] for label in str_labels]

    for idx, (name, (_, y)) in enumerate(data.items()):
        label_counts = Counter([int(label) for label in y])
        counts = [label_counts.get(label, 0) for label in str_labels]

        r = [x + barWidth * idx for x in range(len(counts))]
        ax.bar(r, counts, width=barWidth, color=colors[idx], align='center', label=name)

    # Adding labels
    ax.set_xlabel('Labels', fontweight='bold')
    ax.set_xticks([r + barWidth * (len(data) / 2) for r in range(len(str_labels))])
    ax.set_xticklabels(labels_str, rotation=45, ha='right')
    ax.set_ylabel('Count', fontweight='bold')
    ax.set_title('Label Distribution in Datasets')
    ax.legend()

    plt.tight_layout()

    return fig
