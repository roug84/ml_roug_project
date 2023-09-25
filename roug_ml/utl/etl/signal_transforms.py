import numpy as np


def integrate_signal_with_moving_window(signal: np.ndarray,
                                        window_width: int) -> np.ndarray:
    """
    Perform moving window integration on a signal. This function applies moving window integration
    to the provided signal.
    Parameters
    ----------
    :param: signal : The input signal as a NumPy array.
    :param: signal window_width : The width of the moving window.
    :returns: The integrated signal as a NumPy array.
    """
    # It calculates the cumulative sum of the signal,
    cumsum_signal = np.cumsum(signal)
    # then subtracts the cumulative sum at each point by the cumulative sum at the point
    # `window_width` steps before it.
    integrated_signal = np.zeros_like(signal)
    integrated_signal[window_width:] = cumsum_signal[window_width:] - cumsum_signal[:-window_width]
    return integrated_signal

