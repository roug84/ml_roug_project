import os

import mlflow
import numpy as np
import pandas as pd
from beartype.typing import List, Dict, Tuple
import torch

from roug_ml.utl.evaluation.eval_utl import calc_loss_acc_val
from roug_ml.utl.mlflow_utils import get_best_run, get_or_create_experiment
from roug_ml.utl.paths_utl import create_dir
from roug_ml.models.hyperoptimization import get_best_run_from_hyperoptim
from roug_ml.utl.parameter_utils import generate_param_grid_with_different_size_layers
from roug_ml.utl.processing.filters import LowPassFilter
from roug_ml.utl.processing.signal_processing import SignalProcessor
from roug_ml.utl.transforms.features_management import FeatureFlattener
from roug_ml.utl.set_seed import set_seed
from sklearn.svm import LinearSVC
from roug_ml.models.feature_selection import ModelBasedOneHotSelector
from sklearn.linear_model import LogisticRegression

set_seed(42)

from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif

from roug_ml.models.hyperoptimization import parallele_hyper_optim
from roug_ml.utl.parameter_utils import restructure_dict
from roug_ml.utl.processing.dataset_processing import covert_3a_from_pandas_to_dict
from roug_ml.utl.etl.data_extraction import extract_activity_data_from_users
from roug_ml.models.pipelines.pipelines import NNTorch
from projects.har.configs import M_HEALTH_PATH
from roug_ml.bases import MLPipeline
from roug_ml.utl.etl.transforms_utl import one_hot_to_numeric, integer_to_onehot


def filter_data_dict(data_dict: dict, subjects: list) -> dict:
    """
    Filter the data_dict dictionary based on a specific set of subjects.
    :param data_dict: The input dictionary.
    :param subjects: The list of subjects to keep.
    returns: The filtered dictionary.
    """
    filtered_dict = {
        key: [value[i] for i in range(len(data_dict['user'])) if data_dict['user'][i] in subjects]
        for key, value in data_dict.items()}
    return filtered_dict


class MHealtAnalysis(MLPipeline):

    def __init__(self, in_mlflow_experiment_name) -> None:
        """
        Analysis deta from
        """
        super().__init__(in_mlflow_experiment_name)

        self.list_of_positions = ['chest', 'ankle', 'right_arm']
        # self.list_of_positions = ['right_arm']
        self.number_of_point_per_activity = 511  # TODO: automatic

        self.features_to_extract = ['mean', 'std', 'rms', 'max', 'min', 'var']

        # Define the number of parallel workers
        # Use all available CPU cores except one
        self.num_workers = 1  # multiprocessing.cpu_count() - 1
        self.framework = 'torch'
        # self.framework = 'tf'

        # model params:
        self.nn_params_keys = ["activations", "in_nn", "input_shape", "output_shape"]
        self.other_keys = ["batch_size", "cost_function", "learning_rate", "metrics", "n_epochs",
                           "nn_key"]

        self.results_path = os.path.join(M_HEALTH_PATH, '../../roug_ml/models')
        create_dir(self.results_path)

        self.setup_mlflow()

        self.re_optimize = True
        print(self.mlflow_experiment_id)

        self.clf = None

    def setup_mlflow(self):
        mlflow.set_tracking_uri("http://localhost:8000")
        # mlflow.set_tracking_uri('http://host.docker.internal:8000')
        self.mlflow_experiment_id = get_or_create_experiment(self.mlflow_experiment_name)

    def run(self):
        """
        # 1. Load datasets
        # 2. Process data to right format
        # 3. Create in_dataset and split
        # 4. Optimize and get best run or Get the best run from mlopt
        # 5. Validate
        """
        # 1. Load data
        dataset = self.collect_data(M_HEALTH_PATH)

        # 2. Preprocess data to right format
        final_data_set = self.preprocess_data(in_dataset=dataset)
        key_position = 'x_' + 'right_arm'
        # key_position = 'x_' + 'chest'
        # key_position = 'x_' + 'ankle'
        final_data_set = self.extract_features(final_data_set,
                                               in_key=key_position,
                                               in_features_to_extract=self.features_to_extract,
                                               sample_freq=50,
                                               window_size=1,
                                               cutoff_frequency=5
                                               )

        # 3b. Split by patients: in the original in_dataset we use
        x_train, y_train, x_val, y_val = self.split_data(final_data_set=final_data_set)

        # plot_label_distribution_from_arrays(label_mapping=M_HEALTH_ACTIVITIES_LABELS,
        #                                     Train=(x_train, np.argmax(y_train, axis=1)),
        #                                     Validation=(x_val, np.argmax(y_val, axis=1))
        #                                     )

        # self.clf = LogisticRegression(max_iter=10000,
        #                               class_weight='balanced'
        #                               ).fit(x_train, one_hot_to_numeric(y_train))

        pipeline = Pipeline(steps=[
            ('feature_selection', SelectKBest(f_classif, k=130)),
            ('estimator', LogisticRegression(max_iter=10000, class_weight='balanced'))])
        self.clf = pipeline.fit(x_train, one_hot_to_numeric(
            y_train))  # LogisticRegression(max_iter=10000,
        # class_weight='balanced'
        # ).fit(X_train, one_hot_to_numeric(y_train_labels))

        if self.re_optimize:
            # 4. Optimize and get best run
            results = self.hyperoptimize(x_train, y_train, x_val, y_val)

            best_params, best_val_accuracy, best_run_id = get_best_run_from_hyperoptim(results)
        else:
            # 4. Get the best run from mlopt
            best_run_id, best_params = get_best_run(self.mlflow_experiment_name, "val_accuracy")

            best_params = restructure_dict(best_params, self.nn_params_keys, self.other_keys,
                                           in_from_mlflow=True)

        # 5. Validate
        self.validate(x_val, y_val, best_run_id, best_params)

    @staticmethod
    def extract_features(final_data_set: dict,
                         in_key: str,
                         in_features_to_extract: list,
                         sample_freq: int,
                         window_size: float,
                         cutoff_frequency: int):
        """
        Prepare the data by generating the x_data and y_hot_data arrays.
        :param final_data_set: The final data set containing the input data.
        :param in_key: The key to access the input data in the final_data_set.
        :param in_features_to_extract: A list of features to extract from the data. These features
                are extracted from final_data_set[in_key] and used as input (x_train).
        :param sample_freq: The sample frequency in Hz.
        :param window_size: The size of the window in seconds.
        :param cutoff_frequency: The cutoff frequency for low-pass filtering.

        Returns:
            tuple: A tuple containing the x_train and y_train_oh arrays.
        """

        sampling_rate = 1 / sample_freq
        points_for_mean = int(window_size / sampling_rate)

        dataset_features = []
        for x in final_data_set[in_key]:
            sp = SignalProcessor(input_signal=x,
                                 in_filter=(LowPassFilter, {'cutoff_frequency': cutoff_frequency}),
                                 in_window_size=points_for_mean,
                                 in_stride=points_for_mean
                                 )
            sp.apply_filter(in_signal=sp.input_signal)
            # sp.calibrate_data(in_signal=sp.output_signal)
            features = sp.extract_windowed_features(in_signal=sp.output_signal)
            dataset_features.append(features)

        flattener = FeatureFlattener(final_data_set, in_features_to_extract)
        new_feat = flattener.run(dataset_features)

        final_data_set['new_feat'] = np.asarray(new_feat)

        return final_data_set

    def split_data(self, final_data_set: Dict
                   ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Splits a dataset into training and validation sets. The function selects subjects for
        training and validation sets, then reshapes the feature vectors to match the expected input
        shape of the model.

        :param final_data_set: The final dataset to be split. The dictionary keys should be
        'new_feat' and 'y', representing feature vectors and their corresponding labels,
        respectively. Each subject's data should be indexed by 'subject#' in this dictionary.

        returns:
        x_train: The training feature vectors reshaped for the model's expected input.
        y_train: The training labels corresponding to the training features.
        x_val: The validation feature vectors reshaped for the model's expected input.
        y_val: The validation labels corresponding to the validation features.
        """
        subjects_to_keep = ['subject' + str(i) for i in range(7)]
        data_set_train = filter_data_dict(final_data_set, subjects_to_keep)

        subjects_to_keep = ['subject' + str(i + 7) for i in range(3)]
        data_set_test = filter_data_dict(final_data_set, subjects_to_keep)

        x_val = np.asarray(data_set_test['new_feat'])
        y_val_oh = np.asarray(data_set_test['y'])

        x_train = np.asarray(data_set_train['new_feat'])
        y_train_oh = np.asarray(data_set_train['y'])

        # Reshape the input data to match the expected shape of the model
        x_train = x_train.reshape(x_train.shape[0], x_train.shape[1])
        x_val = x_val.reshape(x_val.shape[0], x_val.shape[1])

        y_train = y_train_oh
        y_val = y_val_oh

        return x_train, y_train, x_val, y_val

    @staticmethod
    def collect_data(in_path: str) -> pd.DataFrame:
        """
        read data
        :param in_path: the path
        """
        _, _, filenames = next(os.walk(in_path))
        total_df = []
        for file_x in filenames:
            if file_x.endswith(".log"):
                print(file_x[8:-4])
                path_to_read = os.path.join(in_path, file_x)
                #  read file path_to_read
                input_df = pd.read_csv(path_to_read, delimiter="\t", header=None)
                input_df[24] = file_x[8:-4]
                total_df.append(input_df[[0, 1, 2, 5, 6, 7, 14, 15, 16, 23, 24]])
        return pd.concat(total_df, axis=0)

    def preprocess_data(self, in_dataset: pd.DataFrame):
        """
        Preprocesses the raw dataset for the analysis.

        This method performs the following steps:
        1. Extracts data for chest, ankle, and right arm by converting the input dataset into
           dictionaries.
        2. Consolidates the generated data into a single dictionary.

        3. Create segments of data of the same size

        :param in_dataset: pandas DataFrame containing the raw data to be preprocessed.

        :return: A dictionary containing the preprocessed data. The dictionary keys are:
        - 'y_num': Numeric labels for the activities.
        - 'y_onehot': One-hot encoded labels for the activities.
        - 'x_chest': Sensor readings from the chest.
        - 'x_ankle': Sensor readings from the ankle.
        - 'x_right_arm': Sensor readings from the right arm.
        - 'User': User identifiers.
        """
        # 1. Extracts data from dataset
        dataset_dict_chest = covert_3a_from_pandas_to_dict(in_dataset=in_dataset,
                                                           in_label_col=23,
                                                           in_user_col=24,
                                                           in_3a_col=[0, 1, 2])

        dataset_dict_ankle = covert_3a_from_pandas_to_dict(in_dataset=in_dataset,
                                                           in_label_col=23,
                                                           in_user_col=24,
                                                           in_3a_col=[5, 6, 7])

        dataset_dict_right_arm = covert_3a_from_pandas_to_dict(in_dataset=in_dataset,
                                                               in_label_col=23,
                                                               in_user_col=24,
                                                               in_3a_col=[14, 15, 16])
        # 2. Single dict
        dataset_dict = {
            'y_num': dataset_dict_chest['y_num'],
            'y_onehot': dataset_dict_chest['y_onehot'],
            'x_chest': dataset_dict_chest['x'],
            'x_ankle': dataset_dict_ankle['x'],
            'x_right_arm': dataset_dict_right_arm['x'],
            'User': dataset_dict_chest['User']
        }

        # 3. Extract data from each PA
        final_data_set = self.extract_data_during_activity(
            in_num_of_users=10,
            in_dataset_dict=dataset_dict,
            in_accelerometer_position=self.list_of_positions)

        # 4. Segment of same size
        final_data_set = self.segment_data(
            in_data=final_data_set, in_points_in_each_set=self.number_of_point_per_activity,
            in_accelerometer_position=self.list_of_positions)

        return final_data_set

    @staticmethod
    def _split_train_test(data: np.array, in_points_in_each_set: int):
        """
        Splits the data into a training set and a test set.

        Parameters:
        data (np.array): The data to be split.
        in_points_in_each_set (int): Number of points in each in_dataset.

        Returns:
        tuple: Contains arrays for training data and test data.
        """
        # Convert 3a to image for each position
        train_data = data[0:in_points_in_each_set]
        test_data = data[in_points_in_each_set + 1: 2 * in_points_in_each_set + 1]
        return train_data, test_data

    def extract_data_during_activity(self,
                                     in_num_of_users: int = 10,
                                     in_dataset_dict: Dict = None,
                                     in_accelerometer_position: List[str] = None) -> Dict:
        """
        Extract data from physical activities periods (Label > 0) from in_dataset.
        :param in_num_of_users: number of patients
        :param in_dataset_dict: dictionary with data
        :param in_accelerometer_position: position of accelerometer in the body
        :return: dictionary final_data_set with train test data.
        """

        data_keys = [f"x_{key}" for key in in_accelerometer_position]
        final_data_set = {key: [] for key in ['y_label', 'y_onehot', 'user', *data_keys]}

        list_of_classes = in_dataset_dict['y_num']
        for label in np.unique(list_of_classes):
            # Number of users
            list_user = ['subject' + str(i + 1) for i in range(in_num_of_users)]
            for user_i in list_user:
                user_data_dict = extract_activity_data_from_users(
                    user_i, label, in_dataset_dict, self.list_of_positions)

                for key in in_accelerometer_position:
                    final_data_set[f"x_{key}"].append(user_data_dict[f"{key}_label"])

                final_data_set['y_label'].append(label)
                final_data_set['y_onehot'].append(user_data_dict['y_onehot'][0])
                final_data_set['user'].append(user_i)
        return final_data_set

    def split_data_for_train_test_portion_of_pa(self,
                                                in_data: Dict,
                                                in_points_in_each_set: int = int(1022 / 2),
                                                in_accelerometer_position=None) -> Dict:
        """
        Split data for training and test sets. in_points_in_each_set for training and
        in_points_in_each_set for testing. Data is already a dict of lists where x[0] is a set of 3a
        date that correspond to label y_label[0] and its one_hot_version (y_onehot[0]). Therefore,
        the only variable that is processed (split) here is x
        :param in_data: dictionary with data
        :param in_points_in_each_set: int, number of points in the in_dataset
        :param in_accelerometer_position: position of accelerometer in the body
        :return: dictionary with train test data.
        """
        # TODO: split is done 50% of an activity period trainning and 50% of the same activity for
        #  training. Improve this

        if in_accelerometer_position is None:
            in_accelerometer_position = ['x_chest', 'x_ankle', 'x_right_arm']

        train_test_keys = [f"x_{key}_train" for key in in_accelerometer_position] + \
                          [f"x_{key}_test" for key in in_accelerometer_position]
        final_data_set = {key: [] for key in ['y_label', 'y_onehot', 'user', *train_test_keys]}

        for key in in_accelerometer_position:
            for data in in_data[f"x_{key}"]:
                train_data, test_data = self._split_train_test(data, in_points_in_each_set)
                final_data_set[f"x_{key}_train"].append(train_data)
                final_data_set[f"x_{key}_test"].append(test_data)

        # Labels are the same training and test set
        final_data_set['y_label'] = in_data['y_label']
        final_data_set['y_onehot'] = in_data['y_onehot']
        final_data_set['user'] = in_data['user']
        final_data_set['y'] = final_data_set['y_onehot']

        # List of keys to keep:
        base_keys = ['y', 'y_label', 'y_onehot', 'user']
        keys_to_keep = base_keys + [f"x_{key}_train" for key in in_accelerometer_position]

        # Create a new dictionary with only these keys as follows:
        data_set_train = {k: final_data_set[k] for k in keys_to_keep if k in final_data_set}

        # List of keys to keep:
        keys_to_keep = base_keys + [f"x_{key}_test" for key in in_accelerometer_position]

        # Create a new dictionary with only these keys as follows:
        data_set_test = {k: final_data_set[k] for k in keys_to_keep if k in final_data_set}

        return data_set_train, data_set_test

    def segment_data(self,
                     in_data: Dict,
                     in_points_in_each_set: int = int(1022/2),
                     in_accelerometer_position=None) -> Dict:
        """
        Cut data of each PA into segments of the same size
        Data is already a dict of lists where x[0] is a set of 3a
        date that correspond to label y_label[0] and its one_hot_version (y_onehot[0]). Therefore,
        the only variable that is processed (split) here is x
        :param in_data: dictionary with data
        :param in_points_in_each_set: int, number of points in the in_dataset
        :param in_accelerometer_position: position of accelerometer in the body
        :return: dictionary with train test data.
        """
        # TODO: split is done 50% of an activity period trainning and 50% of the same activity for
        #  training. Improve this

        if in_accelerometer_position is None:
            in_accelerometer_position = ['x_chest', 'x_ankle', 'x_right_arm']

        train_test_keys = [f"x_{key}" for key in in_accelerometer_position]
        final_data_set = {key: [] for key in ['y_label', 'y_onehot', 'user', *train_test_keys]}

        for key in in_accelerometer_position:
            for data in in_data[f"x_{key}"]:
                train_data, test_data = self._split_train_test(data, in_points_in_each_set)
                final_data_set[f"x_{key}"].append(train_data)

        # Labels are the same training and test set
        final_data_set['y_label'] = in_data['y_label']
        final_data_set['y_onehot'] = in_data['y_onehot']
        final_data_set['user'] = in_data['user']
        final_data_set['y'] = final_data_set['y_onehot']

        # List of keys to keep:
        base_keys = ['y', 'y_label', 'y_onehot', 'user']
        keys_to_keep = base_keys + [f"x_{key}" for key in in_accelerometer_position]

        # Create a new dictionary with only these keys as follows:
        data_set_train = {k: final_data_set[k] for k in keys_to_keep if k in final_data_set}

        return data_set_train

    def generate_params(self, x_train):
        """
        This function generates the list of parameters for the hyperparameter optimization.

        :return: list_params: list of dictionaries, where each dictionary contains a unique
         combination of hyperparameters
        """

        # Define hyperparameters
        nn_key = ['CNN',
                  # 'MLP'
                  ]
        input_shape = [np.asarray(x_train).shape[1]]
        output_shape = [13]
        batch_size = [#1,
                      32#, 64,
             # 128
                      ]
        cost_function = [torch.nn.CrossEntropyLoss()]
        learning_rate = [#0.01,
            0.001]
        n_epochs = [10,
                    15, 20,
                    # 100
            ]
        metrics = ['accuracy']
        layer_sizes = [
            # [200, 300, 100], #[200, 300, 400, 100], [50, 300, 100, 50],
            #  [100, 200, 200, 100], [50, 100, 100, 100, 50],
            #  [100, 200, 100, 200, 100],
                       [200, 300]
        ]
        activations = [
             # ['sigmoid', 'relu', 'tanh'],
             # ['tanh', 'tanh', 'tanh', 'tanh'], ['relu', 'relu', 'relu', 'relu'],
             # ['relu', 'relu', 'relu', 'relu', 'tanh'],
             # ['relu', 'relu', 'relu', 'relu', 'relu'],
             ['identity', 'relu']
        ]
        cnn_filters = [1,
                       3#, 5, 10
                       ]

        list_params = generate_param_grid_with_different_size_layers(nn_key, input_shape,
                                                                     output_shape, batch_size,
                                                                     cost_function, learning_rate,
                                                                     n_epochs, metrics,
                                                                     layer_sizes, activations,
                                                                     cnn_filters)

        list_params = [
            restructure_dict(params, self.nn_params_keys, self.other_keys, in_from_mlflow=False) for
            params in list_params]

        return list_params

    def hyperoptimize(self, x_train, y, x_val, y_val):
        """
        This function performs hyperparameter optimization using parallel computing.

        :param x_train: array-like, shape (n_samples, n_features), input training data
        :param y: array-like, shape (n_samples, ), target training values
        :param x_val: array-like, shape (n_samples, n_features), input validation data
        :param y_val: array-like, shape (n_samples, ), target validation values
        combination of hyperparameters

        :return: results: list of tuples, each containing the parameters, validation accuracy, and
         run ID for one run of the model
        """
        list_params = self.generate_params(x_train=x_train)
        # Perform hyperparameter optimization
        results = parallele_hyper_optim(in_num_workers=self.num_workers,
                                        x_train=x_train,
                                        y=y,
                                        x_val=x_val,
                                        y_val=y_val,
                                        param_grid_outer=list_params,
                                        in_framework=self.framework,
                                        model_save_path=self.results_path,
                                        in_mlflow_experiment_id=self.mlflow_experiment_id,
                                        use_kfold=False,
                                        in_scaler=None,
                                        # in_selector=AutoencoderFeatureSelector(
                                        #     encoding_dim=100, epochs=10, batch_size=32,
                                        #     learning_rate=0.001, device='cpu')
                                        # in_selector=SelectKBestOneHotSelector(SelectKBest(f_classif,
                                        #                                               k=100)),
                                        in_selector=ModelBasedOneHotSelector(LinearSVC(penalty="l1",
                                                                          dual=False,
                                                                          random_state=42)
                                                                          )

                                        )

        return results

    def validate(self, x_val: np.ndarray, y_val: np.ndarray,
                 best_run_id: str, best_params: Dict) -> None:
        """
        Validates a model using the validation data.
        The function loads the best model obtained during the training phase, then uses it to make
        predictions on the validation data. It then calculates the validation accuracy and the
        confusion matrix.

        :param x_val: The validation feature vectors reshaped for the model's expected input.
        :param y_val: The validation labels corresponding to the validation features.
        :param best_run_id: The ID of the best training run, used to load the best model from MLflow
        :param best_params: The best hyperparameters used to train the model.

        Prints:
        val_acc (float): The validation accuracy.

        Computes:
        Confusion matrix for the given predictions and targets using the defined activity labels.
        """
        # best_model = mlflow.pytorch.load_model("runs:/{}/models".format(best_run_id))
        #
        pipeline_torch = Pipeline(steps=[('NN', NNTorch(**best_params))])
        # pipeline_torch.named_steps['NN'].nn_model = best_model
        print(best_params)
        pipeline_torch = mlflow.sklearn.load_model("runs:/{}/pipeline".format(best_run_id))

        predictions = pipeline_torch.predict(x_val)

        predictions_clf = self.clf.predict(x_val)
        predictions_clf_one_hot = integer_to_onehot(data_integer=predictions_clf,
                                                    n_labels=y_val.shape[1])

        #
        # val_acc = calc_loss_acc_val(predictions, y_val)
        # loaded_pipeline = mlflow.sklearn.load_model("runs:/{}/pipeline".format(best_run_id))
        # predictions = loaded_pipeline.predict(x_val.reshape(x_val.shape[0], x_val.shape[1]))
        # predictions = loaded_pipeline.predict(x_val)
        label_subset = list(range(y_val.shape[1]))  # replace with the labels you are interested in

        # Create a mask for the subset of interest
        mask = np.isin(one_hot_to_numeric(y_val), label_subset)

        # Filter based on the mask
        filtered_y_val_labels = y_val[mask]
        filtered_predictions = predictions[mask]
        filtered_predictions_clf = predictions_clf_one_hot[mask]

        filtered_predictions = integer_to_onehot(data_integer=filtered_predictions,
                                                 n_labels=y_val.shape[1])

        val_acc = calc_loss_acc_val(filtered_predictions, filtered_y_val_labels)
        print(val_acc)

        val_acc_clf = calc_loss_acc_val(filtered_predictions_clf, filtered_y_val_labels)
        print("Val torch: {}".format(val_acc))
        print("Val rdn forest: {}".format(val_acc_clf))

        # compute_multiclass_confusion_matrix(targets=filtered_y_val_labels,
        #                                     outputs_list=[filtered_predictions, filtered_predictions_clf],
        #                                     model_names=['NN', 'rnd_forest'],
        #                                     class_labels=M_HEALTH_ACTIVITIES_LABELS
        #                                     )



        import shap

        x_val = pipeline_torch[0].transform(x_val)
        print(x_val.shape)
        # # Extracting the nn and the scaler
        nn_model = pipeline_torch[-1].nn_model

        # Initialize the SHAP explainer
        explainer = shap.Explainer(nn_model, x_val)

        # Compute SHAP values for a single instance from the validation set
        instance_idx = 0
        shap_values = explainer(x_val[instance_idx:instance_idx+1])

        # # Get feature names
        # feature_names = data.feature_names
        #
        # # Plot the SHAP values for the selected instance
        # shap.plots.waterfall(shap_values[0], max_display=10)# feature_names=feature_names)
        print(shap_values.shape)
        shap.plots.waterfall(shap_values[0, :, 1], max_display=10)

        nn_model.eval()  # Ensure your model is in evaluation mode
        x_val_tensor = torch.tensor(x_val, dtype=torch.float32)

        # Convert your data to PyTorch tensors
        print(x_val_tensor.shape)

        x_val_numpy = x_val_tensor.numpy()
        subset_x_val = x_val_numpy[:10]  # Taking only 10 samples
        print(subset_x_val.shape)

        subset_tensor = x_val_tensor[:10]
        print(subset_tensor.shape)
        subset_tensor = x_val_tensor[:10].unsqueeze(1)

        print(subset_tensor.shape)

        output = nn_model(subset_tensor)
        print(output.shape)

        explainer = shap.DeepExplainer(nn_model, subset_tensor)
        shap_values = explainer.shap_values(subset_tensor)

        print(np.array(shap_values).shape)
        print(subset_tensor.shape)

        # For example, to plot SHAP values for the first class:
        class_idx = 0
        shap_values_for_class = shap_values[class_idx]
        print(shap_values_for_class.shape)  # Should be (10, 1, 126)

        # Sum over the channel dimension to get shape (10, 126)
        shap_values_for_class = shap_values_for_class.sum(axis=1)
        print(shap_values_for_class.shape)  # Should be (10, 126)

        # Plot
        shap.summary_plot(shap_values_for_class, subset_tensor.squeeze(1))

        # scaler = pipe[0]
        #
        # # Keep only some points to limit computational burden
        # max_points = 500
        # factor = int(len(dataset_test["x"]) / max_points)
        # x_test = dataset_test["x"][range(0, len(dataset_test["x"]), factor)]
        # x_test_rescaled = scaler.transform(x_test)
        #
        # explainer = shap.DeepExplainer(nn_model, x_test_rescaled)
        # shap_values = explainer.shap_values(x_test_rescaled)
        #
        # self._shap_values_plot(shap_values, x_test, workflow)


if __name__ == '__main__':
    analysis = MHealtAnalysis(in_mlflow_experiment_name='HARAnalysis_vF_si_fold2')
    analysis.run()
