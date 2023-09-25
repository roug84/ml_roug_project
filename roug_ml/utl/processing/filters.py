import numpy as np
import scipy.stats as stats
from scipy.fftpack import fft
import pandas as pd
from scipy import signal


class LowPassFilter:
    """
    Class to apply a low-pass filter to the input signal data.
    This class applies a low-pass filter to 'signal_data' using the Butterworth filter design
    method from SciPy's signal module.
    A low-pass filter allows signals with a frequency lower than a certain cutoff frequency to pass
    through and blocks signals with frequencies higher than the cutoff frequency.
    """
    def __init__(self, cutoff_frequency: float, sampling_rate: float = 100.0):
        """
        :param cutoff_frequency: Cutoff frequency for the low-pass filter.
        :param sampling_rate: data.
        """
        self.cutoff = cutoff_frequency
        self.sampling_rate = sampling_rate

    def run(self, signal_data: np.ndarray) -> np.ndarray:
        """
        Apply the low-pass filter to the input signal data.

        :param signal_data: Input signal data.

        :returns: filtered_data: The filtered version of the input signal data.
        """
        # Calculate the Nyquist frequency, which is half of the sampling rate.
        nyquist_frequency = 0.5 * self.sampling_rate

        # Normalize the cutoff frequency by dividing it by the Nyquist frequency.
        normalized_cutoff = self.cutoff / nyquist_frequency

        # Use the Butterworth filter design method to get the filter coefficients.
        # Here, '4' is the order of the filter, 'normalized_cutoff' is the
        # normalized cutoff frequency, 'low' means it's a low-pass filter, and 'analog=False'
        # means it's a digital filter.
        b, a = signal.butter(4, normalized_cutoff, btype='low', analog=False)

        # Apply the filter to the input signal data.
        filtered_data = signal.filtfilt(b, a, signal_data, axis=0)

        return filtered_data


class DataCalibrator:
    """
    Class to calibrate input signal data.
    This class calibrates the input signal data by subtracting the mean of the signal data
    (calibration offset) from each data point. Calibration is used to adjust the signal data so that
    its mean is zero, this helps in removing any constant offset from the signal.
    """
    def __init__(self):
        self.calibration_offsets = None

    def run(self, signal_data: np.ndarray) -> np.ndarray:
        """
        Calibrate the input signal data.

        :param signal_data: Input numpy array of signal data.

        :returns: calibrated_data: A numpy array of calibrated signal data.
        """
        # Compute the mean of the signal data, this will serve as the calibration offset.
        self.calibration_offsets = np.mean(signal_data, axis=0)

        # Subtract the calibration offset from the signal data to get the calibrated data.
        calibrated_data = signal_data - self.calibration_offsets

        return calibrated_data
