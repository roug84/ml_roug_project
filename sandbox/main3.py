import numpy as np
from os import walk
import pandas as pd
import os

from processing.dataset_processing import covert_3a_from_pandas_to_dict

from roug_ml.utl.etl.extract_utl import compute_windowed_features, \
    apply_low_pass_filter, flatting_array_consecutive
from etl.data_extraction import extract_activity_data_from_users
import tensorflow as tf

from sklearn.pipeline import Pipeline
from models.pipelines.pipelines import NNTensorFlow
from configs.data_paths import M_HEALTH_PATH

np.random.seed(1)
tf.random.set_seed(2)


class MHealtAnalysis:
    def __init__(self) -> None:
        """
        Performs following analysis:
            - Extract data
            - Make plots for analysis
        """
        self.list_of_positions = ['chest', 'ankle', 'right_arm']
        # self.list_of_positions = ['right_arm']
        self.number_of_point_per_activity = 511  # TODO: automatic

    def run(self):
        """


        """
        # Load data
        dataset = self.read_data(M_HEALTH_PATH)

        # Process data to right format
        dataset_dict_chest = covert_3a_from_pandas_to_dict(dataset=dataset,
                                                           in_label_col=23,
                                                           in_user_col=24,
                                                           in_3a_col=[0, 1, 2])

        dataset_dict_ankle = covert_3a_from_pandas_to_dict(dataset=dataset,
                                                           in_label_col=23,
                                                           in_user_col=24,
                                                           in_3a_col=[5, 6, 7])

        dataset_dict_right_arm = covert_3a_from_pandas_to_dict(dataset=dataset,
                                                               in_label_col=23,
                                                               in_user_col=24,
                                                               in_3a_col=[14, 15, 16])

        dataset_dict = {
            'y_num': dataset_dict_chest['y_num'],
            'y_onehot': dataset_dict_chest['y_onehot'],
            'x_chest': dataset_dict_chest['x'],
            'x_ankle': dataset_dict_ankle['x'],
            'x_right_arm': dataset_dict_right_arm['x'],
            'User': dataset_dict_chest['User']
        }

        # Create in_dataset
        final_data_set = self.extract_data_during_activity(
            in_num_of_users=10,
            in_dataset_dict=dataset_dict,
            in_accelerometer_position=self.list_of_positions)

        data_set_train, data_set_test = self.split_data_for_train_test(
            in_data=final_data_set, in_points_in_each_set=self.number_of_point_per_activity,
            in_accelerometer_position=self.list_of_positions)

        x_train, y_train_oh = self.create_x_y(data_set_train, in_key='x_right_arm_train')

        X_val, _ = self.create_x_y(data_set_test, in_key='x_right_arm_test')

        # final_data_set = extract_features(final_data_set, in_list_positions=['right_arm', 'chest',
        # 'ankle'])

        # x_test = np.asarray(final_data_set['x_ankle_test'])
        # y_test_oh = np.asarray(final_data_set['y'])

        # params2 = {
        #     'nn_key': 'AE',
        #     'nn_params': {
        #         'input_shape': final_data_set['x_feat'][0].shape[1],
        #         'output_shape': final_data_set['x_feat'][0].shape[1],
        #         'in_nn': [400, 1000, 10000, 1000, 400],
        #         # 'activations': ['relu', 'relu', 'relu', 'relu']
        #         'activations': ['tanh', 'tanh', 'tanh', 'tanh', 'tanh']
        #     },
        #     'batch_size': 10,
        #     'cost_function': 'mse',
        #     'learning_rate': 0.001,
        #     'n_epochs': 100,
        #     'metrics': ['accuracy']}
        #
        # # Create pipeline
        # # model = Pipeline(steps=[('RobustScaler', StandardScaler()), ('NN', NNTensorFlow(**params2))])
        # model = Pipeline(steps=[('NN', NNTensorFlow(**params2))])
        #
        # X_val = np.asarray(final_data_set['x_feat_test'])
        # X = np.asarray(final_data_set['x_feat'])
        # y = y_train_oh.reshape(130, 13)
        # params_pipeline = {}
        # params_pipeline[model.steps[-1][0] + "__validation_data"] = (X_val.reshape(X_val.shape[0], X_val.shape[2]),
        #                                                              X_val.reshape(X_val.shape[0], X_val.shape[2]))
        # model.fit(X.reshape(X.shape[0], X.shape[2]),
        #           X.reshape(X.shape[0], X.shape[2]), **params_pipeline)
        #
        # plt.plot(model.predict(X.reshape(X.shape[0], X.shape[2])))
        # plt.plot(X.reshape(X.shape[0], X.shape[2]))
        # plt.show()
        #
        # y_pred = model.predict(X_val.reshape(X_val.shape[0], X_val.shape[2]))
        # acc = compute_accuracy_from_soft_max(y=y, y_pred=y_pred)
        # print(acc)
        # model.evaluate(np.asarray(final_data_set['x_feat']), y_train_oh.reshape(130, 1, 13))
        # model.evaluate(np.asarray(final_data_set['x_feat_test']), y_train_oh.reshape(130, 1, 13))

        params2 = {
            'nn_key': 'MLP',
            'nn_params': {
                'input_shape': np.asarray(x_train).shape[1],
                'output_shape': 13,
                'in_nn': [200, 300],
                'activations': ['relu', 'relu']
            },
            'batch_size': 10,
            'cost_function': 'categorical_crossentropy',
            'learning_rate': 0.001,
            'n_epochs': 36,
            'metrics': ['accuracy']}

        # Create pipeline
        # model = Pipeline(steps=[('RobustScaler', StandardScaler()), ('NN', NNTensorFlow(**params2))])

        # model = NNTensorFlow(**params2)
        # model.load('/Users/hector/DiaHecDev/results/Models_DL', 'servingpa')
        # X_val = x_train#np.asarray(final_data_set['x_feat_test'])
        # X = np.asarray(final_data_set['x_feat'])
        y = y_train_oh.reshape(130, 13)

        model = Pipeline(steps=[('NN', NNTensorFlow(**params2))])
        params_pipeline = {}
        params_pipeline[model.steps[-1][0] + "__validation_data"] = (
        X_val.reshape(X_val.shape[0], X_val.shape[1]), y)
        model.fit(X_val.reshape(X_val.shape[0], X_val.shape[1]), y, **params_pipeline)

        # params_pipeline = {}
        # params_pipeline[model.steps[-1][0] + "__validation_data"] = (X_val.reshape(X_val.shape[0], X_val.shape[2]), y)
        # model.fit(X.reshape(X.shape[0], X.shape[2]), y, **params_pipeline)
        # # model.fit(X.reshape(X.shape[0], X.shape[2]), y, validation_data=(X_val.reshape(X_val.shape[0], X_val.shape[2]), y))
        # # model.save('/Users/hector/DiaHecDev/results/Models_DL', 'servingpa')
        # # mlflow.sklearn.save_model(model, "my_model")
        # # mlflow.tensorflow.save_model(model, "my_model")
        # y_pred = model.predict(X_val.reshape(X_val.shape[0], X_val.shape[2]))
        # acc = compute_accuracy_from_soft_max(y=y, y_pred=y_pred)
        # print(acc)
        #
        # # Log a parameter (key-value pair)
        # for k, v in params2.items():
        #     # print(k, v)
        #     log_param(k, v)
        #
        # # Log a metric; metrics can be updated throughout the run
        # log_metric("acc", acc)
        # log_artifacts('models', model)

        # # Log an artifact (output file)
        # if not os.path.exists("outputs"):
        #     os.makedirs("outputs")
        # with open("outputs/test.txt", "w") as f:
        #     f.write("hello world!")
        # log_artifacts("outputs")
        # model.evaluate(np.asarray(final_data_set['x_feat']), y_train_oh.reshape(130, 1, 13))
        # model.evaluate(np.asarray(final_data_set['x_feat_test']), y_train_oh.reshape(130, 1, 13))
        #
        #
    def create_x_y(self, final_data_set, in_key):
        sample_freq = 50  # Hz
        sampling_rate = 1 / sample_freq
        windo_size = 1  # [s]
        points_for_mean = int(windo_size / sampling_rate)
        # windowed_feats = compute_windowed_features(
        #     in_features=final_data_set['x_right_arm_train'][0],
        #     in_window_size=points_for_mean,
        #     in_stride=points_for_mean)
        CUTOFF_FREQUENCY = 5
        dataset_features = []
        for x in final_data_set[in_key]:
            # x = unflatting_array_consecutive(x, size=3)
            # plt.plot(x)
            # plt.show()
            low_pass_filtered_data = apply_low_pass_filter(x, CUTOFF_FREQUENCY)
            # calibrated_data = calibrate_data(low_pass_filtered_data)
            features = compute_windowed_features(in_features=low_pass_filtered_data,
                                                 in_window_size=points_for_mean,
                                                 in_stride=points_for_mean)
            dataset_features.append(features)
        dataset_features = {key: [feat[key] for feat in dataset_features] for key in
                            dataset_features[0].keys()}

        for key in ['mean', 'std', 'rms', 'max', 'min', 'var']:
            final_data_set[key + 'flat'] = np.asarray(
                np.array(
                    [flatting_array_consecutive(np.asarray(x)) for x in dataset_features[key]]))

        new_feat = []
        for i in range(len(final_data_set['meanflat'])):
            big_feat = []
            for keyi in ['meanflat', 'stdflat', 'rmsflat', 'maxflat', 'minflat', 'varflat']:
                # print(keyi)
                big_feat.append(final_data_set[keyi][i])
                # plt.plot(dataset_train2[keyi][i])
                # plt.show()
            new_feat.append(np.asarray(np.concatenate(big_feat)))
        final_data_set['new_feat'] = np.asarray(new_feat)

        # final_data_set = extract_features(final_data_set,
        #                                   in_list_positions=['right_arm'])

        #
        # x_train = np.asarray(final_data_set['meanflat'])
        x_train = final_data_set['new_feat']
        y_train_oh = np.asarray(final_data_set['y'])
        return x_train, y_train_oh

    @staticmethod
    def read_data(in_path: str):
        """
        read data
        :param in_path: the path
        """
        _, _, filenames = next(walk(in_path))
        total_df = []
        for file_x in filenames:
            if file_x.endswith(".log"):
                print(file_x[8:-4])
                # Jutar preprocessed_path \ file_x
                path_to_read = os.path.join(in_path, file_x)
                #  leer archivo path_to_read
                input_df = pd.read_csv(path_to_read, delimiter="\t", header=None)
                input_df[24] = file_x[8:-4]
                total_df.append(input_df[[0, 1, 2, 5, 6, 7, 14, 15, 16, 23, 24]])
        return pd.concat(total_df, axis=0)

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
                                     in_dataset_dict: dict = {},
                                     in_accelerometer_position: list = ['chest', 'ankle',
                                                                        'right_arm']) -> dict:
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

    def split_data_for_train_test(self,
                                  in_data: dict,
                                  in_points_in_each_set: int = int(1022 / 2),
                                  in_accelerometer_position=None) -> dict:
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

        # Label are the same training and test set
        final_data_set['y_label'] = in_data['y_label']
        final_data_set['y_onehot'] = in_data['y_onehot']
        final_data_set['user'] = in_data['user']
        final_data_set['y'] = final_data_set['y_onehot']

        # And you have a list of keys that you want to keep:
        base_keys = ['y', 'y_label', 'y_onehot', 'user']
        keys_to_keep = base_keys + [f"x_{key}_train" for key in in_accelerometer_position]

        # You can create a new dictionary with only these keys as follows:
        data_set_train = {k: final_data_set[k] for k in keys_to_keep if k in final_data_set}

        keys_to_keep = base_keys + [f"x_{key}_test" for key in in_accelerometer_position]

        # You can create a new dictionary with only these keys as follows:
        data_set_test = {k: final_data_set[k] for k in keys_to_keep if k in final_data_set}

        return data_set_train, data_set_test


if __name__ == '__main__':
    analysis = MHealtAnalysis()
    analysis.run()
