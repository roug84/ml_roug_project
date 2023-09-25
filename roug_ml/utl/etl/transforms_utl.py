import numpy as np
import os
import ntpath
import pandas as pd


def integer_to_onehot(data_integer: np.array, n_labels: int = None):
    """
    Function to convert integer to onehot
    :param data_integer: labels in int format
    :param n_labels: number of labels
    """
    if n_labels:
        data_onehot = np.zeros(shape=(data_integer.shape[0], n_labels))
    else:
        data_onehot = np.zeros(shape=(data_integer.shape[0], data_integer.max() + 1))
    for row in range(data_integer.shape[0]):
        integer = int(data_integer[row])
        data_onehot[row, integer] = 1
    return data_onehot


def one_hot_to_numeric(one_hot_array: np.ndarray) -> np.ndarray:
    """
    Convert a one-hot encoded array back to its numeric representation.

    :param one_hot_array: The one-hot encoded data.
    :returns numeric representation of the one-hot data.
    """
    return np.argmax(one_hot_array, axis=1)



# 'x_chest': x_chest, 'x_ankle': x_ankle, 'x_rightarm
def create_train_test_img_datasets(num_of_samples, dataset_dict, activities_label, number_of_points_to_plot=200,
                                   points_for_training=int(1022 / 2)):
    """
    act_duration: Duration of the PA
    """
    one_hot_labels_list = []
    x_chest_train_list = []
    x_ankle_train_list = []
    x_rightarm_train_list = []

    x_chest_train_img_list = []
    x_ankle_train_img_list = []
    x_rightarm_train_img_list = []

    x_chest_test_img_list = []
    x_ankle_test_img_list = []
    x_rightarm_test_img_list = []

    activity_name_list = []
    y_label_list = []
    user_list = []
    list_of_acts = dataset_dict['y_num']
    for label in np.unique(list_of_acts):
        list_user = ['subject' + str(i + 1) for i in range(num_of_samples)]
        for user_i in list_user:
            # Select data for user
            cond_user = dataset_dict['User'] == user_i

            # Select for label
            cond_label = dataset_dict['y_num'] == label

            # get data of user for a given activity in the 3 positions
            x_chest_label = dataset_dict['x_chest'][cond_label & cond_user]
            x_ankle_label = dataset_dict['x_ankle'][cond_label & cond_user]
            x_rightarm_label = dataset_dict['x_rightarm'][cond_label & cond_user]

            # get label in categorical and one hot version
            y_label = dataset_dict['y_num'][cond_label & cond_user]
            y_onehot = dataset_dict['yhot'][cond_label & cond_user]

            # Convert 3a to image for each position
            x_chest_img = x_chest_label[0:points_for_training].reshape(x_chest_label[0:points_for_training].shape[0], 3,
                                                                       1)

            x_chest_img_test = x_chest_label[points_for_training + 1: 2 * points_for_training + 1].reshape(
                x_chest_label[points_for_training + 1: 2 * points_for_training + 1].shape[0], 3, 1)

            x_ankle_img = x_ankle_label[0:points_for_training].reshape(x_ankle_label[0:points_for_training].shape[0], 3,
                                                                       1)

            x_ankle_img_test = x_ankle_label[points_for_training + 1: 2 * points_for_training + 1].reshape(
                x_ankle_label[points_for_training + 1: 2 * points_for_training + 1].shape[0], 3, 1)


            x_rightarm_img = x_rightarm_label[0:points_for_training].reshape(
                x_rightarm_label[0:points_for_training].shape[0], 3, 1)

            x_rightarm_img_test = x_rightarm_label[points_for_training + 1: 2 * points_for_training + 1].reshape(
                x_rightarm_label[points_for_training + 1: 2 * points_for_training + 1].shape[0], 3, 1)

            # Normalize
            x_chest_img = x_chest_img  # / 30.
            x_chest_img_test = x_chest_img_test  # / 30.
            x_ankle_img = x_ankle_img  # / 30.
            x_ankle_img_test = x_ankle_img_test  # / 30.
            x_rightarm_img = x_rightarm_img  # / 30.
            x_rightarm_img_test = x_rightarm_img_test# / 30.

            # Append data to lists that will be converted on dictionary
            one_hot_labels_list.append(y_onehot[0])
            x_chest_train_list.append(x_chest_label)
            x_ankle_train_list.append(x_ankle_label)
            x_rightarm_train_list.append(x_rightarm_label)
            x_chest_train_img_list.append(x_chest_img)
            x_ankle_train_img_list.append(x_ankle_img)
            x_rightarm_train_img_list.append(x_rightarm_img)
            x_chest_test_img_list.append(x_chest_img_test)
            x_ankle_test_img_list.append(x_ankle_img_test)
            x_rightarm_test_img_list.append(x_rightarm_img_test)
            activity_name_list.append(activities_label[label])
            y_label_list.append(label)
            user_list.append(user_i)
            """
            fig, ax = plt.subplots(figsize=(10, 1))
            #ax.imshow(x_chest_img[0:200, :, :].reshape(3, 200), cmap='Greys', vmin=-1, vmax=1)
            plt.plot(x_ankle_label[0:number_of_points_to_plot])
            plt.plot(x_chest_label[0:number_of_points_to_plot])
            plt.plot(x_rightarm_label[0:number_of_points_to_plot])
            plt.title(M_HEALTH_ACTIVITIES_LABELS[label] + ' ' + user_i)
            #plt.plot(y_label, 'b--')
            plt.show()
            """
    # Create datasets dictionary
    final_data_set = {'x': x_chest_train_img_list,
                      'y': one_hot_labels_list,
                      'x_chest_img_train': x_chest_train_img_list,
                      'x_ankle_img_train': x_ankle_train_img_list,
                      'x_rightarm_img_train': x_rightarm_train_img_list,
                      'x_chest_train': x_chest_train_list,
                      'x_ankle_train': x_ankle_train_list,
                      'x_rightarm_train': x_rightarm_train_list,
                      'x_chest_img_test': x_chest_test_img_list,
                      'x_ankle_img_test': x_ankle_test_img_list,
                      'x_rightarm_img_test': x_rightarm_test_img_list,
                      'y_label': y_label_list,
                      'y_hot': one_hot_labels_list,
                      'user': user_list,
                      }

    # dataset_dict['x_chest_img'] = np.asarray(x_chest_train_img_list)
    # dataset_dict['x_ankle_img'] = np.asarray(x_ankle_train_img_list)
    # dataset_dict['x_rightarm_img'] = np.asarray(x_rightarm_train_img_list)

    return final_data_set

def convert_to_3d_list_of_key_in_dict(final_data_set,
                                      list_of_key_to_3d=['x_chest_train', 'x_ankle_train', 'x_rightarm_train',
                                                         'x_chest_test', 'x_ankle_test', 'x_rightarm_test']):
    """
    act_duration: Duration of the PA
    """
    new_dict = {}
    for key, value in final_data_set.items():
        if key in list_of_key_to_3d:
            new_dict[key + '_3d'] = [x.reshape(x.shape[0], 1, x.shape[1]) for x in final_data_set[key]]
    # new = final_data_set.update(new_dict)
    return {**final_data_set, **new_dict}


def extract_data(in_path, one_hot_labels_list, x_list, y_label_list, user_list, y_num_list,
                 activity_name_list):
    for dirname, _, filenames in os.walk(in_path):
        for filename in filenames:
            tmp_path = os.path.join(dirname, filename)
            print(os.path.join(dirname, filename))
            df_data = pd.read_csv(tmp_path)
            user = ntpath.basename(tmp_path)[:-4]
            df_data['y_num'] = 0
            df_data.loc[df_data['StartHesitation'] == 1, 'y_num'] = 1
            df_data.loc[df_data['Turn'] == 1, 'y_num'] = 2
            df_data.loc[df_data['Walking'] == 1, 'y_num'] = 3

            df_data = add_identificator_of_change_in_variable(df_data, in_var='y_num')
            df_data['user'] = user

            for label in df_data['state_period_number'].unique():
                # Select data for user
                df_pat_tmp = df_data.loc[df_data['state_period_number'] == label]
                if len(df_pat_tmp['y_num']) < 200:
                    continue
                x_list.append(np.asarray(df_pat_tmp[['AccV', 'AccML', 'AccAP']].iloc[0:200]))
                one_hot_labels_list.append(integer_to_onehot(df_pat_tmp['y_num'].iloc[0:200].to_numpy(), n_labels=4))
                user_list.append(user)
                y_num_list.append(df_pat_tmp['y_num'].iloc[0:200].to_numpy())
            del df_pat_tmp, df_data
    return one_hot_labels_list, x_list, y_label_list, user_list, y_num_list, activity_name_list


def add_identificator_of_change_in_variable(df_data, in_var):
    """

    """
    df_data.loc[df_data[in_var] < 1, 'state_number'] = 0
    df_data.loc[df_data[in_var] >= 1, 'state_number'] = 1

    pa_df_patient = df_data[['state_number']].copy()
    df_data = df_data.drop(['state_number'], axis=1)

    pa_df_patient['change'] = pa_df_patient['state_number'].diff()

    df_tmp_2 = pa_df_patient.loc[pa_df_patient['state_number'] == 1, 'change'].copy()
    pa_df_patient['state_period_number'] = df_tmp_2.loc[~df_tmp_2.index.duplicated()].cumsum(axis=0)
    df_data = df_data.join(pa_df_patient, how='outer')
    # df_data['state_period_number'].plot()
    return df_data





