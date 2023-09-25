from scipy.stats import kurtosis, skew
from scipy.fftpack import fft
from beartype.typing import List
import numpy as np
import pandas as pd

ONE_MINUTE_IN_SECOND = 60
ONE_SECOND_IN_MILLISECONDS = 60


class WindowedFeatureComputer:
    """
    Class to compute a variety of statistical and spectral features on a windowed basis.
    """
    def __init__(self, in_window_size: int, in_stride: int = 1):
        """
        Initialize the class with window size and stride.
        """
        self.in_window_size = in_window_size
        self.in_stride = in_stride

    def run(self, in_features: np.ndarray):
        """
        Function to compute a variety of statistical and spectral features on a windowed basis.

        :parameters in_features: Input numpy array from which features are to be computed.

        :returns features: A dictionary where each key is a feature name and the corresponding value
         is af list of the computed feature for each window.
        """
        # Initialize the list of dictionaries.
        dict_patients = []

        # Iterate over the features array using a sliding window.
        for i in range(0, len(in_features) - self.in_window_size, self.in_stride):
            # Initialize a dictionary to store features for the current window.
            feat = {}

            # Obtain the current window from the features array.
            window = in_features[i:i + self.in_window_size]

            # Compute statistical features and store them in the features dictionary.
            feat['mean'] = np.nanmean(window, axis=0)
            feat['std'] = np.nanstd(window, axis=0)
            feat['rms'] = np.sqrt(np.mean(np.square(window), axis=0))
            feat['max'] = np.max(window, axis=0)
            feat['min'] = np.min(window, axis=0)
            feat['var'] = np.var(window, axis=0)
            feat['kurtosis'] = kurtosis(window, axis=0)
            feat['skew'] = skew(window, axis=0)

            # Compute norm-based features and store them in the features dictionary.
            feat['euclidean_norm'] = np.mean(np.linalg.norm(window, axis=1))
            feat['l1_norm'] = np.mean(np.linalg.norm(window, ord=1, axis=1))

            # Compute frequency domain metrics and store them in the features dictionary.
            feat['fft'] = np.abs(fft(window, axis=0))
            feat['fft_energy'] = np.asarray(np.sum(feat['fft'] ** 2, axis=0) / len(feat['fft']))
            feat['max_magnitude_fft'] = np.max(feat['fft'], axis=0)

            # Compute FFT for the first dimension of the window and store it in the features
            # dictionary.
            feat['fft_x1'] = fft(window[:, 0])

            # Append the features dictionary to the list of dictionaries.
            dict_patients.append(feat)

        # Organize the features into a dictionary where each key is a feature name and the
        # corresponding value is a list of the computed feature for each window.
        features = {key: [feat[key] for feat in dict_patients] for key in dict_patients[0].keys()}

        # Convert some features from list to numpy array.
        features['fft_energy'] = np.asarray(features['fft_energy'])
        features['max_magnitude_fft'] = np.asarray(features['max_magnitude_fft'])
        features['fft'] = np.asarray(features['fft'])
        return features


def create_regression_matrix(in_df: pd.DataFrame,
                             in_lags: int,
                             columns_to_lag: List[str],
                             prediction_minutes_ahead_list: List[int],
                             sampling_time_minutes: int = 5
                             ) -> pd.DataFrame:
    """
    Creates a regression matrix with lagged features for specified columns and multiple prediction
    ahead times.

    :param in_df: the input data with a column "time" and other specified columns.
    :param in_lags: the number of lag values you want to consider.
    :param columns_to_lag: the columns for which you want to create lag features.
    :param prediction_minutes_ahead_list: the number of minutes for each future
    prediction.
    :param sampling_time_minutes: the time in minutes for each sampling step.
    :return: regression matrix with lagged features and multiple target columns.
    """
    # Set the time column as index
    in_df.set_index('time', inplace=True)

    # Generate a full date range for the time span of the dataset
    full_date_range = pd.date_range(start=in_df.index.min(), end=in_df.index.max(),
                                    freq=f"{sampling_time_minutes}T")

    # Reindex the dataframe using the full date range, this will introduce NaN values for missing
    # times
    in_df = in_df.reindex(full_date_range)

    df_lagged = in_df.copy()
    for col in columns_to_lag:
        for i in range(1, in_lags + 1):
            df_lagged[f"{col}_lag_{i}"] = df_lagged[col].shift(i)

    for minutes_ahead in prediction_minutes_ahead_list:
        # Calculate the number of rows to shift for the prediction
        prediction_shift = minutes_ahead // sampling_time_minutes

        # Add predicted values column shifted backward for each prediction_minutes_ahead value
        df_lagged[f"gly_value_predicted_{minutes_ahead}min_ahead"] = \
            df_lagged['gly_value'].shift(-prediction_shift)

    df_lagged = df_lagged.dropna().reset_index()
    df_lagged.rename(columns={'index': 'time'}, inplace=True)
    return df_lagged


if __name__ == '__main__':
    # Usage:
    df = pd.DataFrame({
        'time': pd.to_datetime(
            ['2022-01-01 00:05', '2022-01-01 00:10', '2022-01-01 00:15',
             '2022-01-01 00:20',
             '2022-01-01 00:25', '2022-01-01 00:30', '2022-01-01 00:35',
             '2022-01-01 00:40'
             ]),
        'gly_value': [1, 2, 3,
                      4,
                      4,
                      5, 6,
                      7
                      ],
        'insulin': [5, 6, 7,
                    8,
                    1, 2, 3,
                    4
                    ]
    })

    lags = 3
    columns = ['gly_value', 'insulin']
    prediction_list = [5, 10, 20]
    regression_matrix = create_regression_matrix(df, lags, columns, prediction_list)
    print(regression_matrix)
    print(regression_matrix)
    # For the first matrix:
    matrix_1 = regression_matrix[
        ['gly_value', 'gly_value_lag_1', 'gly_value_lag_2', 'insulin_lag_1', 'insulin_lag_2']]

    # For the second matrix:
    matrix_2 = regression_matrix[['gly_value_predicted_5min_ahead']]

    matrix_1_values = matrix_1.values
    matrix_2_values = matrix_2.values









