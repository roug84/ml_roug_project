from roug_ml.utl.processing.filters import LowPassFilter, DataCalibrator
from roug_ml.utl.processing.feature_extration import WindowedFeatureComputer


class SignalProcessor:
    def __init__(self,
                 input_signal,
                 in_filter=(LowPassFilter, {'cutoff_frequency': 0.3}),
                 in_window_size=10,
                 in_stride=2):
        self.input_signal = input_signal
        self.output_signal = None
        self.filter = None
        self.sampler = None
        self.transform = None
        self.amplifier = None
        self.filter = in_filter[0](in_filter[1]['cutoff_frequency'])
        self.calibrator = DataCalibrator()

        self.window_size = in_window_size
        self.stride = in_stride

    def apply_filter(self, in_signal):
        # Apply the filter to the input_signal and save the result in output_signal
        self.output_signal = self.filter.run(in_signal)

    def calibrate_data(self, in_signal):
        # Calibrate the signal (subtract the mean)
        self.output_signal = self.calibrator.run(in_signal)

    def extract_windowed_features(self, in_signal):
        feature_computer = WindowedFeatureComputer(in_window_size=self.window_size,
                                                   in_stride=self.stride)
        return feature_computer.run(in_signal)

    def sample(self, sampler):
        # Sample the input_signal using the sampler
        pass

    def apply_transform(self, transform):
        # Apply the transform to the input_signal and save the result in output_signal
        pass

    def amplify(self, amplifier):
        # Amplify the input_signal using the amplifier
        pass

