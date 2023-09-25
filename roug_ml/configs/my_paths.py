import os

root_path = os.path.abspath(os.path.join(__file__, os.pardir))

# Path to main data folder
data_path = os.path.abspath(os.path.join(root_path, '../../../data'))

# path to kaggle folder
KAGGLE_ECG_PATH = os.path.join(data_path, 'kaggle')

# Path to results
RESULTS_PATH = os.path.abspath(os.path.join(root_path, '../../../results'))

# path to mlflow folder
MLFLOW_BACK_PATH = os.path.join(data_path, 'artifacts')
