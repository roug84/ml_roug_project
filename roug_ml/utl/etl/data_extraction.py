def extract_activity_data_from_users(user_i: str, label: int, in_dataset_dict: dict,
                                     in_position: list):
    """
    Extracts data for a specific user and activity from the in_dataset.
    param: user_i: Identifier for the user.
    param: label: The label of the activity.
    param: in_dataset_dict: The dataset dictionary.
    param: in_points_in_each_set: Number of points in each dataset.
    param: in_position: positions of 3a in the body
    Returns:
    tuple: Contains arrays for chest, ankle, and right arm data, and their corresponding labels
    and one-hot encodings.
    """
    # Select data for user
    cond_user = in_dataset_dict['User'] == user_i

    # Select for label
    cond_label = in_dataset_dict['y_num'] == label

    # Get data of user for a given activity in in_position
    labels = {}
    for key in in_position:
        labels[f"{key}_label"] = in_dataset_dict[f"x_{key}"][cond_label & cond_user]

    # Get label in categorical and one hot version
    labels['y_num'] = in_dataset_dict['y_num'][cond_label & cond_user]
    labels['y_onehot'] = in_dataset_dict['y_onehot'][cond_label & cond_user]

    return labels
