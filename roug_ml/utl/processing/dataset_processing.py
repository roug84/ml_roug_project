import numpy as np
import pandas as pd
from roug_ml.utl.etl.transforms_utl import integer_to_onehot
from beartype.typing import Union, List, Dict


def covert_3a_from_pandas_to_dict(in_dataset: Union[pd.DataFrame, pd.Series],
                                  in_label_col: int = 23,
                                  in_user_col: int = 24,
                                  in_3a_col: List[int] = None) -> Dict:
    """
    Process the input in_dataset by extracting specific elements, converting them to NumPy arrays,
    and storing them in a dictionary. One hot version of labels is computed.
    :param in_dataset: data with the following elements are used:
    :param in_label_col: number of column with labels (type of activity)
    :param in_user_col: number of column with user id
    :param in_3a_col: list of length 3 corresponding to x, y ,z respectively
    """
    dict_data = {
        'y_num': np.asarray(in_dataset[in_label_col]),  # in_label_col is the type of activity
        # One-hot encoded array of the activity type.
        'y_onehot': integer_to_onehot(np.asarray(in_dataset[in_label_col])),
        # NumPy array of the chest data.
        'x': np.asarray(in_dataset[in_3a_col]),
        # NumPy array of the user.
        'User': np.asarray(in_dataset[in_user_col])
    }
    return dict_data
