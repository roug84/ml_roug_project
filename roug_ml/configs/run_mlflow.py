# run_mlflow.py
import os
from roug_ml.configs.my_paths import MLFLOW_BACK_PATH

# Construct the mlflow server command with the desired artifact path
mlflow_command = f"mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root {MLFLOW_BACK_PATH}/artifacts --host 0.0.0.0 --port 8000"

# Execute the command
os.system(mlflow_command)