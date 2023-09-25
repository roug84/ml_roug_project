import numpy as np


def integer_to_onehot(data_integer: np.array,
                      n_labels: int,
                      start_at_zero: bool = True) -> np.array():
    """
    Function to convert integer categorical data to one-hot encoding.
    param: data_integer: A numpy array of integer type that we want to convert to one-hot encoding.
    param: n_labels: The number of classes in 'data_integer'. If provided, the one-hot encoding will
    have 'n_labels' columns.
    param: start_at_zero: A boolean parameter to determine whether the labels in 'data_integer' start
     from 0 or 1.
           If 'True', it's assumed that the labels start from 0.
           If 'False', it's assumed that the labels start from 1, and the function adjusts
            the indices accordingly.
    """

    # Initialize a 2D numpy array 'data_onehot' with zeros. Its number of rows is the same as
    # 'data_integer'.
    # The number of columns is determined by 'n_labels' if it's provided, otherwise, it's determined
    # by the maximum value in 'data_integer' + 1.
    if n_labels:
        data_onehot = np.zeros(shape=(data_integer.shape[0], n_labels))
    else:
        data_onehot = np.zeros(shape=(data_integer.shape[0], data_integer.max() + 1))

    # Loop over the rows of 'data_integer'.
    for row in range(data_integer.shape[0]):
        # Convert the current integer to an index.
        integer = int(data_integer[row])

        # If labels in 'data_integer' don't start at zero, adjust the index by subtracting 1.
        if not start_at_zero:
            integer = integer - 1

        # Set the corresponding position in 'data_onehot' to 1.
        data_onehot[row, integer] = 1
    return data_onehot
