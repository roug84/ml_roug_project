import os

from roug_ml.configs.my_paths import data_path, RESULTS_PATH

# path to TCGA Pan cancer
TCGA_DATA_PATH = os.path.join(data_path, 'pancan-gtex-target')

# path to save results related to this project
TCGA_RESULTS_PATH = os.path.join(RESULTS_PATH, 'tcga')
