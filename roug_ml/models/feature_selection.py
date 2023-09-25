
"""

# Example usage of RFE
svc = LinearSVC(penalty="l1", dual=False, random_state=42)
selector = RFEFeatureSelector(svc, n_features_to_select=50)
selector.fit(x_train, y)

# or

# Example usage of Trees
selector = TreeFeatureSelector(n_estimators=100, max_depth=10, threshold=0.05)
selector.fit(x_train, y)

# or

# Example usage of VarianceThreshold
selector = VarianceThresholdSelector(threshold=0.01)
selector.fit(x_train, y)


1. Filter Methods
a. Variance Thresholding
Principle: Removes features with variance below a certain cutoff.
Best when:
There are features with zero or very low variance.
You have many features, and you need a quick way to reduce dimensionality before applying other methods.
b. Correlation Coefficient Methods
Principle: Features highly correlated with the target variable are kept while removing the others.
Best when:
You want to find the linear relationship between features and the target variable.
Useful for linear models, where multicollinearity can be a concern.
2. Wrapper Methods
a. Recursive Feature Elimination (RFE)
Principle: Recursively removes the least important features.
Best when:
You have a moderate number of features.
Computation time is not a major constraint since it's computationally more expensive than filter methods.
You need a ranking of feature importances.
3. Embedded Methods
a. Feature Importance from Trees (like Random Forest)
Principle: Uses the intrinsic properties of tree-based algorithms to determine feature importance.
Best when:
Non-linear relationships exist in the data.
You want to capture interactions between features.
Robustness to outliers is needed.
b. L1 Regularization (Lasso)
Principle: Adds a penalty to the absolute values of coefficients, which can drive some feature coefficients to zero.
Best when:
You're dealing with linear relationships and you want feature selection as part of the regularization process.
You have many features, some of which might be irrelevant.
Tips to choose the right method:
Understand the Data: It's crucial to have a basic understanding of the dataset's characteristics and the relationships it might contain (linear, non-linear).

Dimensionality: For very high-dimensional datasets, start with filter methods or L1 regularization as they're computationally more efficient.

Model Type: If you're going to use linear models, ensure the selected features don't have multicollinearity. For tree-based models, embedded methods like feature importances from trees are more straightforward.

Computation Time: Wrapper methods (like RFE) tend to be more computationally expensive than filter or embedded methods.

Stability: Some methods, especially tree-based ones, might give different importances if there's a slight change in the dataset. In such cases, techniques like recursive feature elimination with cross-validation (RFECV) can provide more stability.

Combining Methods: It's not uncommon to combine multiple feature selection methods. For instance, one might use variance thresholding to remove zero-variance features and then apply RFE or L1 regularization.

In conclusion, feature selection is as much an art as it is science. Often, it's beneficial to try multiple methods and validate the model's performance on a held-out set or through cross-validation to determine the best feature selection approach.

"""

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import RFE
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import SelectFromModel
from sklearn.svm import LinearSVC

from sklearn.base import BaseEstimator, TransformerMixin


class SelectKBestOneHotSelector(BaseEstimator, TransformerMixin):
    """
    A feature selector that wraps around the SelectKBest method, tailored for data that may be one-hot encoded.

    :param selector: The SelectKBest selector instance.
    """

    """
    Feature selection for one-hot encoded labels.

    :param selector: Feature selector that uses score functions.
    """

    def __init__(self, selector):
        self.selector = selector

    def fit(self, x: np.ndarray, y: np.ndarray = None) -> BaseEstimator:
        """
        Fit the selector using the provided data.

        :param x: Training data
        :param y: One-hot encoded labels
        :return: self
        """
        y_labels = np.argmax(y, axis=1)
        self.selector.fit(x, y_labels)
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        """
        Transform the data by selecting the relevant features.

        :param x: Data to be transformed
        :return: Transformed data
        """
        return self.selector.transform(x)

    def get_support(self, indices=False):
        """
        Get a mask, or integer index, of the features selected

        :param indices: If True, the method will return indices instead of a boolean mask.
        :return: Boolean mask or list of indices of the selected features
        """
        return self.selector.get_support(indices=indices)


class ModelBasedOneHotSelector(BaseEstimator, TransformerMixin):
    """
    Feature selection based on a given estimator. Specifically designed for one-hot encoded labels.

    :param estimator: Estimator to use for feature selection.
    """
    def __init__(self, estimator):
        self.estimator = estimator
        self.selector = SelectFromModel(self.estimator)

    def fit(self, x: np.ndarray, y: np.ndarray = None) -> BaseEstimator:
        """
        Fit the selector using the provided data.

        :param x: Training data
        :param y: One-hot encoded labels
        :return: self
        """
        y_labels = np.argmax(y, axis=1)
        self.selector.fit(x, y_labels)
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        """
        Transform the data by selecting the relevant features.

        :param x: Data to be transformed
        :return: Transformed data
        """
        return self.selector.transform(x)

    def get_support(self, indices=False):
        """
        Get a mask, or integer index, of the features selected

        :param indices: If True, the method will return indices instead of a boolean mask.
        :return: Boolean mask or list of indices of the selected features
        """
        return self.selector.get_support(indices=indices)


class TreeFeatureSelector(BaseEstimator, TransformerMixin):
    """
    Feature selection based on feature importance from RandomForestClassifier.

    :param n_estimators: int, The number of trees in the forest.
    :param max_depth: int, Maximum depth of the tree.
    :param threshold: float, Threshold value to select features. If None, mean importance is used.
    """

    def __init__(self, n_estimators: int = 100,
                 max_depth=None, threshold=None):
        self.estimator = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth)
        self.threshold = threshold

    def fit(self, x: np.ndarray, y: np.ndarray or None = None) -> BaseEstimator:
        """
        Fit the selector using the provided data.

        :param x: Training data
        :param y: One-hot encoded labels
        :return: self
        """
        y_labels = np.argmax(y, axis=1)
        self.estimator.fit(x, y_labels)
        self.importances = self.estimator.feature_importances_
        if self.threshold is None:
            self.threshold = np.mean(self.importances)
        self.indices = np.where(self.importances > self.threshold)[0]
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        """
        Transform the data by selecting the relevant features based on their importance.

        :param x: Data to be transformed
        :return: Transformed data
        """
        return x[:, self.indices]


class RFEFeatureSelector(BaseEstimator, TransformerMixin):
    """
    Feature selection using Recursive Feature Elimination (RFE).

    :param estimator: A supervised learning estimator with a fit method.
    :param n_features_to_select: int, Number of top features to select.
    """

    def __init__(self, estimator, n_features_to_select=None):
        self.estimator = estimator
        self.n_features_to_select = n_features_to_select
        self.selector = RFE(self.estimator, n_features_to_select=self.n_features_to_select)

    def fit(self, x: np.ndarray, y: np.ndarray or None = None) -> BaseEstimator:
        """
        Fit the RFE selector.

        :param x: Training data
        :param y: One-hot encoded labels
        :return: self
        """
        y_labels = np.argmax(y, axis=1)
        self.selector.fit(x, y_labels)
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        """
        Transform the data by selecting the relevant features.

        :param x: np.array, Data to be transformed
        :return: np.array, Transformed data
        """
        return self.selector.transform(x)


class VarianceThresholdSelector(BaseEstimator, TransformerMixin):
    """
    Feature selector that removes all low-variance features.

    :param threshold: float, Features with a training-set variance lower than this threshold will be removed.
    """

    def __init__(self, threshold: float = 0.0):
        self.selector = VarianceThreshold(threshold)

    def fit(self, x: np.ndarray, y: np.ndarray or None = None):
        """
        Fit the Variance Threshold selector.

        :param x: Training data
        :return: self
        """
        self.selector.fit(x)
        return self

    def transform(self, x: np.ndarray):
        """
        Transform the data by selecting features based on variance.

        :param x: np.array, Data to be transformed
        :return: np.array, Transformed data
        """
        return self.selector.transform(x)


class Autoencoder(nn.Module):
    """
    Simple Autoencoder model.

    :param input_dim: int, Input dimension of the data.
    :param encoding_dim: int, Dimension of the encoded data.
    """

    def __init__(self, input_dim, encoding_dim):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, encoding_dim),
            nn.ReLU(inplace=True)
        )
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, input_dim),
            nn.Sigmoid()  # If your data is in the range [0,1]. Remove otherwise.
        )

    def forward(self, x: np.ndarray):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


class AutoencoderFeatureSelector(BaseEstimator, TransformerMixin):
    """
    Feature selector based on Autoencoder.

    :param encoding_dim: int, Dimension of the encoded data.
    :param epochs: int, Number of training epochs.
    :param batch_size: int, Training batch size.
    :param learning_rate: float, Learning rate for optimization.
    :param device: str, Device for training ('cpu' or 'cuda').
    """

    def __init__(self, encoding_dim, epochs=10, batch_size=32, learning_rate=0.001, device='cpu'):
        self.encoding_dim = encoding_dim
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.device = device
        self.autoencoder = None

    def fit(self, x: np.ndarray, y=None):
        """
        Train the autoencoder for feature selection.

        :param x: np.array, Training data
        :return: self
        """
        input_dim = x.shape[1]
        self.autoencoder = Autoencoder(input_dim, self.encoding_dim).to(self.device)

        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.autoencoder.parameters(), lr=self.learning_rate)

        x_tensor = torch.FloatTensor(x).to(self.device)

        for epoch in range(self.epochs):
            optimizer.zero_grad()
            outputs = self.autoencoder(x_tensor)
            loss = criterion(outputs, x_tensor)
            loss.backward()
            optimizer.step()

        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        """
        Transform the data to its encoded form.

        :param x: Data to be transformed
        :return: Transformed (encoded) data
        """
        self.autoencoder.eval()  # Set to evaluation mode
        x_tensor = torch.FloatTensor(x).to(self.device)
        encoded = self.autoencoder.encoder(x_tensor).detach().cpu().numpy()
        return encoded

# Example usage:
# selector = AutoencoderFeatureSelector(encoding_dim=10, device='cuda')
# selector.fit(x_train)
# x_train_transformed = selector.transform(x_train)

