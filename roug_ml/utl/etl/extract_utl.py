import numpy as np
import scipy.stats as stats
from scipy.fftpack import fft
import pandas as pd
from scipy import signal


def add_identificator_of_change_in_variable(df_data: pd.DataFrame, in_var: str) -> pd.DataFrame:
    """
    Function to add a column indicating a state change in a variable.
    This function modifies the input DataFrame to add an identifier of change for a given variable.
    The new column 'state_number' is added based on whether the variable value is less than 1
    (state 0) or  greater than or equal to 1 (state 1). Then a column 'change' is added to indicate
    when 'state_number' changes.bAnother column 'state_period_number' is added to mark cumulatively
    each period where 'state_number' is 1.
    :param df_data: Input pandas DataFrame.
    :param in_var: Name of the column in df_data for which the change of state should be identified.

    :returns: df_data: Modified pandas DataFrame with additional columns 'state_number', 'change',
    'state_period_number'.
    """

    # Create a new column 'state_number' in the DataFrame.
    # If the value of 'in_var' is less than 1, 'state_number' is set to 0.
    # If the value of 'in_var' is greater than or equal to 1, 'state_number' is set to 1.
    df_data.loc[df_data[in_var] < 1, 'state_number'] = 0
    df_data.loc[df_data[in_var] >= 1, 'state_number'] = 1

    # Create a copy of the DataFrame with only the 'state_number' column.
    pa_df_patient = df_data[['state_number']].copy()

    # Drop the 'state_number' column from the original DataFrame.
    df_data = df_data.drop(['state_number'], axis=1)

    # Create a new column 'change' in pa_df_patient, showing the difference between each row and its
    # previous row.
    pa_df_patient['change'] = pa_df_patient['state_number'].diff()

    # Create a DataFrame df_tmp_2 where 'state_number' equals 1, and keep only the 'change' column.
    df_tmp_2 = pa_df_patient.loc[pa_df_patient['state_number'] == 1, 'change'].copy()

    # Add the column 'state_period_number' to pa_df_patient to mark cumulatively each period where
    # 'state_number' is 1.
    pa_df_patient['state_period_number'] = df_tmp_2.loc[~df_tmp_2.index.duplicated()].cumsum(axis=0)

    # Join the original DataFrame df_data with pa_df_patient on the index, keeping all rows.
    df_data = df_data.join(pa_df_patient, how='outer')

    return df_data


def flatting_array_consecutive(in_array: np.ndarray) -> np.ndarray:
    """
    Function to flatten a numpy array with 'F' (Fortran-like) order.
    This function returns a flattened version of the input array. The 'ravel' function is used with
    'F' (Fortran-like) order, meaning that it flattens the array column-wise. In other words, the
     elements in the same column are considered as consecutive elements.
    :param in_array: Input numpy array.

    :returns: A 1D numpy array, which is a flattened version of the input array.
    """
    return in_array.ravel('F')


def unflatting_array_consecutive(in_array: np.ndarray, size: int = 3) -> np.ndarray:
    """
    Function to reshape a flattened numpy array back into a 2D array.
    This function reshapes the input 1D array into a 2D array of a specified size,
    then transposes the array. The reshape operation follows the 'C' order (row-wise).
    The '-1' in the reshape function allows numpy to calculate the exact dimension for that axis.
    Parameters:
    :param in_array: Input 1D numpy array.
    :param size: The size of the first dimension of the reshaped array.
    :returns: A 2D numpy array, reshaped and transposed version of the input array.
    """
    return in_array.reshape(size, -1).T


# Calibrate the data to remove bias or offset
def calibrate_data(signal_data: np.ndarray) -> np.ndarray:
    """
    Function to calibrate input signal data.
    This function calibrates the input signal data by subtracting the mean of the signal data
    (calibration offset) from each data point. Calibration is used to adjust the signal data so that
    its mean is zero, this helps in removing any constant offset from the signal.

    :param signal_data: Input numpy array of signal data.

    :returns: calibrated_data: A numpy array of calibrated signal data.
    """
    # Compute the mean of the signal data, this will serve as the calibration offset.
    calibration_offsets = np.mean(signal_data, axis=0)

    # Subtract the calibration offset from the signal data to get the calibrated data.
    calibrated_data = signal_data - calibration_offsets

    return calibrated_data


def compute_magnitude(in_data: np.ndarray) -> np.ndarray:
    """
    Function to compute the magnitude (Euclidean norm) of each row in the input data.
    This function can be useful in scenarios with multi-dimensional data like 3D acceleration,
    where the magnitude gives a combined measure of the acceleration in all three directions.

    :param in_data: Input numpy array from which the magnitude is to be computed.

    :returns: magnitudes: A numpy array containing the magnitude of each row of the input data.
    """
    # Compute the Euclidean norm (magnitude) for each row of the input data.
    magnitudes = np.linalg.norm(in_data, axis=1)

    return magnitudes


def count_steps_threshold(
        in_features: np.ndarray,
        in_window_size: int,
        in_threshold: float) -> list:
    """
    Function to detect and count steps based on a threshold.

    This function uses a simple threshold-based algorithm to detect steps in a window of features.
    It iteratively moves the window through the feature array, and in each window,
    if the maximum value is above the given threshold, it counts as a step.
    It also returns additional features like the mean of the window and the number of steps within each window.

    Parameters:
    in_features: Input numpy array from which steps are to be detected.
    in_window_size: The size of the window to use for step detection.
    in_threshold: The threshold above which a maximum value in a window is considered a step.

    Returns:
    dict_patients: A list of dictionaries where each dictionary corresponds to a window of data and contains
    the mean of the window and the number of steps detected in the window.
    """
    # Initialize the steps count and the list of dictionaries.
    steps = 0
    dict_patients = []

    # Iterate over the features array using a sliding window.
    for i in range(len(in_features) - in_window_size):
        # Initialize a dictionary to store features for the current window.
        feat = {}

        # Obtain the current window from the features array.
        window = in_features[i:i + in_window_size]

        # Compute the mean of the window and store it in the features dictionary.
        feat['mean'] = np.nanmean(window)

        # Initialize a variable to count the number of steps in the current window.
        steps_window = 0

        # Iterate over the window and count the number of steps based on the threshold.
        for j in range(len(window)):
            if max(window) > in_threshold:
                steps_window += 1

        # Store the number of steps in the current window in the features dictionary.
        feat['steps_window'] = steps_window

        # If the maximum value in the window is above the threshold, increment the total steps count
        if max(window) > in_threshold:
            steps += 1

        # Append the features dictionary to the list of dictionaries.
        dict_patients.append(feat)

    return dict_patients


def compute_windowed_features(in_features: np.ndarray, in_window_size: int, in_stride: int = 1):
    """
    Function to compute a variety of statistical and spectral features on a windowed basis.

    This function computes a variety of statistical and spectral features for windows of the input
    features.
    It iteratively moves the window through the features array, and in each window, it computes
    several features including mean, standard deviation, RMS, max, min, variance, kurtosis,
    skewness, Euclidean norm, L1 norm, FFT, FFT energy, and maximum FFT magnitude.

    Parameters:
    in_features: Input numpy array from which features are to be computed.
    in_window_size: The size of the window to use for feature computation.
    in_stride: The number of features to move the window by in each iteration. Default is 1.

    Returns:
    features: A dictionary where each key is a feature name and the corresponding value is a list of
    the computed
              feature for each window.
    """
    # Initialize the list of dictionaries.
    dict_patients = []

    # Iterate over the features array using a sliding window.
    for i in range(0, len(in_features) - in_window_size, in_stride):
        # Initialize a dictionary to store features for the current window.
        feat = {}

        # Obtain the current window from the features array.
        window = in_features[i:i + in_window_size]

        # Compute statistical features and store them in the features dictionary.
        feat['mean'] = np.nanmean(window, axis=0)
        feat['std'] = np.nanstd(window, axis=0)
        feat['rms'] = np.sqrt(np.mean(np.square(window), axis=0))
        feat['max'] = np.max(window, axis=0)
        feat['min'] = np.min(window, axis=0)
        feat['var'] = np.var(window, axis=0)
        feat['kurtosis'] = stats.kurtosis(window, axis=0)
        feat['skew'] = stats.skew(window, axis=0)

        # Compute norm-based features and store them in the features dictionary.
        feat['euclidean_norm'] = np.mean(np.linalg.norm(window, axis=1))
        feat['l1_norm'] = np.mean(np.linalg.norm(window, ord=1, axis=1))

        # Compute frequency domain metrics and store them in the features dictionary.
        feat['fft'] = np.abs(fft(window, axis=0))
        feat['fft_energy'] = np.asarray(np.sum(feat['fft'] ** 2, axis=0) / len(feat['fft']))
        feat['max_magnitude_fft'] = np.max(feat['fft'], axis=0)

        # Compute FFT for the first dimension of the window and store it in the features dictionary.
        feat['fft_x1'] = fft(window[:, 0])

        # Append the features dictionary to the list of dictionaries.
        dict_patients.append(feat)

    # Organize the features into a dictionary where each key is a feature name and the corresponding
    # value is a list of the
    # computed feature for each window.
    features = {key: [feat[key] for feat in dict_patients] for key in dict_patients[0].keys()}

    # Convert some features from list to numpy array.
    features['fft_energy'] = np.asarray(features['fft_energy'])
    features['max_magnitude_fft'] = np.asarray(features['max_magnitude_fft'])
    features['fft'] = np.asarray(features['fft'])
    return features
