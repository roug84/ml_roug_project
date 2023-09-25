import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """
    Set the seed for all possible sources of randomness to ensure reproducibility.

    Args:
        seed (int): Seed value for all the random number generators.

    """
    # Seed for Python's random module
    random.seed(seed)
    # Seed for NumPy's random functions
    np.random.seed(seed)
    # Seed for PyTorch's random functions
    torch.manual_seed(seed)

    # Check if CUDA is available to seed CUDA-based random functions
    if torch.cuda.is_available():
        # Seed for PyTorch's CUDA-based random functions
        torch.cuda.manual_seed(seed)
        # Seed for all GPU devices, if there are more than one
        torch.cuda.manual_seed_all(seed)
        # Make CuDNN deterministic to ensure reproducibility
        torch.backends.cudnn.deterministic = True
        # Disable CuDNN benchmark mode
        torch.backends.cudnn.benchmark = False
