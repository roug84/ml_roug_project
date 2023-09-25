import numpy as np


class FeatureFlattener:
    """
    Class to flatten dataset features for further analysis.

    Attributes:
        dataset (dict): A dictionary containing the dataset.
        features_to_extract (list): A list of features to be extracted from the dataset.
        new_features (list): A list to store the newly created big features.
    """

    def __init__(self, dataset, features_to_extract):
        """
        The constructor for FeatureFlattener class.

        :param dataset: The dataset from which the features are to be extracted.
        :param features_to_extract: The list of features to be extracted.
        """
        self.dataset = dataset
        self.features_to_extract = features_to_extract
        self.new_features = []

    def flatten_features(self, dataset_features):
        """
        Method to flatten each feature from the dataset_features dictionary.

        :param dataset_features: A dictionary containing the features and their corresponding values.
        """
        # Iterating through each feature to flatten it and add it to the dataset
        for key in self.features_to_extract:
            self.dataset[key + 'flat'] = np.asarray(
                np.array(
                    [flatting_array_consecutive(np.asarray(x)) for x in dataset_features[key]]))

    def create_big_feature(self):
        """
        Method to create a big feature by concatenating all the flattened features.
        """
        # Iterating through each entry in the dataset
        for i in range(len(self.dataset[self.features_to_extract[0] + 'flat'])):
            big_feat = []
            # Iterating through each feature
            for keyi in [key + 'flat' for key in self.features_to_extract]:
                big_feat.append(self.dataset[keyi][i])
            # Appending the big feature to the new_features list
            self.new_features.append(np.asarray(np.concatenate(big_feat)))

    def run(self, dataset_features):
        """
        The main method to run the feature extraction process.

        :param dataset_features: A list of features extracted from the dataset.
        :return: A list of new features created by concatenating all the flattened features.
        """
        # Formatting the dataset_features into a dictionary
        dataset_features = {key: [feat[key] for feat in dataset_features] for key in
                            dataset_features[0].keys()}

        # Running the two methods defined above
        self.flatten_features(dataset_features)
        self.create_big_feature()

        return self.new_features


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
