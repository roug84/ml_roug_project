from sklearn.model_selection import KFold
from torch.utils.data import DataLoader

from joblib import Parallel, delayed
from sklearn.pipeline import Pipeline
from beartype.typing import List, Dict, Tuple, Union

from roug_ml.models.pipelines.pipelines import NNTorch  #, #, NNTensorFlow

from roug_ml.utl.evaluation.eval_utl import calc_loss_acc_val

import numpy as np
import os
import mlflow
# import tensorflow as tf
import torch
import random
from roug_ml.utl.set_seed import set_seed
from roug_ml.utl.parameter_utils import flatten_dict

os.environ['PYTHONHASHSEED'] = '0'
np.random.seed(123)
random.seed(123)
# tf.random.set_seed(123)

# When using a GPU
os.environ['TF_DETERMINISTIC_OPS'] = '1'
os.environ['TF_CUDNN_DETERMINISTIC'] = '1'

torch.manual_seed(0)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(0)

set_seed(42)


def parallele_hyper_optim(in_num_workers: int,
                          x_train: np.ndarray,
                          y: np.ndarray,
                          x_val: np.ndarray,
                          y_val: np.ndarray,
                          param_grid_outer: List[Dict],
                          in_framework: str,
                          model_save_path: str,
                          in_mlflow_experiment_id: str,
                          use_kfold: bool = False,
                          in_scaler=None,
                          in_selector=None) -> List[Tuple[dict, float]]:
    """
    Perform parallel hyperparameter optimization for a given framework.

    :param in_num_workers: Number of parallel workers.
    :param x_train: Training data.
    :param y: Training labels.
    :param x_val: Validation data.
    :param y_val: Validation labels.
    :param param_grid_outer: Outer parameter grid. This is the hyperparameter space over which
                             the optimization process will search.
    :param in_framework: Framework to use for optimization ('tf' or 'torch').
    :param model_save_path: Path where the trained models will be saved.
    :param in_mlflow_experiment_id: Name of the MLFlow experiment to log the optimization
                                      results.
    :param use_kfold: If True, apply K-Fold Cross Validation during the optimization process.
                      Default is False.

    :return: List of tuples containing optimized parameters and corresponding validation accuracies.
    """

    if in_framework == 'tf':
        # Perform parallel processing for TensorFlow framework
        # return Parallel(n_jobs=in_num_workers)(
        #     delayed(process_params)(params_outer, x_train, y, x_val, y_val)
        #     for params_outer in param_grid_outer
        # )
        return []  # TODO: Fix TF
    else:
        if use_kfold:
            return Parallel(n_jobs=in_num_workers)(
                delayed(kfold_training_with_process_params_torch)(
                    params=params_outer,
                    x_train=x_train,
                    y=y,
                    train_loader=params_outer,
                    k_folds=5,
                    model_save_path=model_save_path,
                    in_mlflow_experiment_name=in_mlflow_experiment_id,
                    in_scaler=in_scaler,
                    in_selector=in_selector)
                for params_outer in param_grid_outer
            )
        else:
            # Perform parallel processing for Torch framework
            # return Parallel(n_jobs=in_num_workers)(
            #     delayed(process_params_torch)(params_outer, x_train, y, x_val, y_val, train_loader,
            #                                   val_loader, model_save_path, in_mlflow_experiment_id
            #                                   )
            #     for params_outer in param_grid_outer
            # )
            results = []
            for params_outer in param_grid_outer:
                result = process_params_torch(params_outer, x_train, y, x_val, y_val,
                                              model_save_path, in_mlflow_experiment_id,
                                              in_scaler=in_scaler, in_selector=in_selector)
                results.append(result)

            # result = process_params_torch(param_grid_outer, x_train, y, x_val, y_val,
            #                               model_save_path, in_mlflow_experiment_id,
            #                               in_scaler=in_scaler, in_selector=in_selector)
            # results.append(result)
            return results


def create_and_fit_pipeline(params: Dict[str, Union[int, str, list]],
                            x_train: np.ndarray,
                            y: np.ndarray,
                            x_val: np.ndarray,
                            y_val: np.ndarray,
                            in_scaler=None,
                            in_selector=None
                            ) -> Tuple:
    """
    Create a pipeline and fit it to the training data.
    :param params: dict, Parameters for the neural network. This should include
        'activations', 'in_nn', 'input_shape', 'output_shape',
        'batch_size', 'cost_function', 'learning_rate',
        'metrics', 'n_epochs', 'nn_key'.
    :param x_train: np.ndarray, The training data.
    :param y: np.ndarray, The training labels.
    :param x_val: np.ndarray, The validation data.
    :param y_val: np.ndarray, The validation labels.
    :param in_scaler: PyTorch DataLoader for the training data.
    :param in_selector: PyTorch DataLoader for the validation data.

    :return: The pipeline created and fitted, and the validation accuracy of the fitted pipeline.
    """
    x_val_transformed = x_val
    steps = []
    # Check if a scaler is provided
    if in_scaler is not None:
        in_scaler.fit(x_train)
        x_train = in_scaler.transforms(x_train)
        steps.append(('scaler', in_scaler))

    # Check if a feature selector is provided
    if in_selector is not None:
        # Fit the feature selector
        in_selector.fit(x_train, y)

        # Get the shape of the transformed data
        x_train = in_selector.transform(x_train)
        new_input_shape = x_train.shape[1]

        # Transforms validation data
        x_val_transformed = in_selector.transform(x_val)

        # Adjust the input_shape NN based
        if isinstance(params, list):
            for i in range(len(params)):
                params[i]['nn_params']['input_shape'] = new_input_shape
        else:
            params['nn_params']['input_shape'] = new_input_shape
        steps.append(('feature_selection', in_selector))

    # add estimator
    #
    if isinstance(params, list):
        torch_stimator = NNTorchEnsemble(params)
    else:
        torch_stimator = NNTorch(**params)
    torch_stimator.fit(x_train, y, **{"validation_data": (x_val_transformed, y_val)})

    steps.append(('estimator', torch_stimator))

    pipeline_torch = Pipeline(steps=steps)

    pipeline_torch.predict(x_val)

    predictions = pipeline_torch.predict(x_val)
    val_acc = calc_loss_acc_val(predictions, y_val)
    return pipeline_torch, val_acc


def process_params_torch(params: Dict[str, Union[int, str, list]],
                         x_train: np.ndarray,
                         y: np.ndarray,
                         x_val: np.ndarray,
                         y_val: np.ndarray,
                         model_save_path: str,
                         in_mlflow_experiment_name: str = 'name',
                         is_fold: bool = False,
                         val_all_folds: float or None = None,
                         in_scaler=None,
                         in_selector=None) -> Tuple[
                                        Dict[str, Union[int, str, list]] or None,
                                        float,
                                        str or None,]:
    """
    Process parameters and compute validation accuracy for Torch pipeline.

    :param params: Parameters for the neural network. This should include
        'activations', 'in_nn', 'input_shape', 'output_shape',
        'batch_size', 'cost_function', 'learning_rate',
        'metrics', 'n_epochs', 'nn_key'.
    :param x_train:The training data.
    :param y: The training labels.
    :param x_val: The validation data.
    :param y_val: The validation labels.
    :param model_save_path: str, The path where the trained model should be saved.
    :param in_mlflow_experiment_name: str, optional, The name of the MLflow experiment where results
        are logged. Default is 'my_experiment_2'.
    :param is_fold: bool, optional, Flag to indicate whether the process is part of a
        cross-validation fold. Default is False.
    :param val_all_folds: float or None, optional, If not None, this accuracy value will override
        the validation accuracy calculated within the function. Default is None.

    :return: A tuple of the processed parameters, validation accuracy, and run ID.
    """
    params = params.copy()

    pipeline_torch, val_accuracy = create_and_fit_pipeline(params, x_train, y, x_val, y_val,
                                                           in_scaler, in_selector)
    # If it is a fold from fold validation then do not save the model to MLflow
    if is_fold:
        return None, val_accuracy, None
    else:
        # if it is not a fold, but it is still k-fold validation then the validation is replaced by
        # the mean of all folds
        if val_all_folds is not None:
            val_accuracy = val_all_folds

        # model_name = 'model_' + '_'.join([f"{k}_{v}" for k, v in params.items()])
        # pipeline_torch.named_steps['NN'].save(model_save_path, model_name)

        with mlflow.start_run(experiment_id=in_mlflow_experiment_name) as run:
            print("Run ID:", run.info.run_id)
            mlflow.log_params(flatten_dict(params))
            mlflow.log_metric("val_accuracy", val_accuracy)
            mlflow.sklearn.log_model(pipeline_torch, "pipeline")
            # mlflow.pytorch.log_model(pipeline_torch.named_steps['NN'].nn_model, "models")
            for k, v in params.items():
                mlflow.log_param(k, v)

        return params, val_accuracy, run.info.run_id


def kfold_training_with_process_params_torch(params: Dict[str, Union[int, str, list]],
                                             x_train: np.ndarray = None,
                                             y: np.ndarray = None,
                                             train_loader=None,
                                             k_folds: int = 5,
                                             model_save_path: str = 'models',
                                             in_mlflow_experiment_name: str = 'name',
                                             in_scaler=None,
                                             in_selector=None
                                             ) -> None:
    """
    Perform k-fold cross-validation training using the `process_params_torch` function.

    :param params: dict, Parameters for the neural network.
    :type params: Dict[str, Union[int, str, list]]
    :param x_train: np.ndarray, Training data.
    :type x_train: np.ndarray, optional
    :param y: np.ndarray, Training labels.
    :type y: np.ndarray, optional
    :param train_loader: Dataset for training.
    :type train_loader: DataLoader, optional
    :param k_folds: Number of folds to use for the cross-validation, defaults to 5
    :type k_folds: int, optional
    :param model_save_path: str, The path where the trained model should be saved.
    :type model_save_path: str, optional
    :param in_mlflow_experiment_name: str, optional, The name of the MLflow experiment where results
     are logged.
    :type in_mlflow_experiment_name: str, optional
    """
    # Assuming X and y are numpy arrays.
    assert isinstance(x_train, np.ndarray)
    assert isinstance(y, np.ndarray)

    kfold = KFold(n_splits=k_folds, shuffle=True)
    vals = []
    for fold, (train_ids, val_ids) in enumerate(kfold.split(x_train)):
        # Sample elements randomly from a given list of ids, no replacement.
        x_train_fold = x_train[train_ids]
        y_train_fold = y[train_ids]

        x_val_fold = x_train[val_ids]
        y_val_fold = y[val_ids]

        # Use the process_params_torch function for training and validation
        processed_params, val_accuracy, run_id = process_params_torch(params,
                                                                      x_train_fold,
                                                                      y_train_fold,
                                                                      x_val_fold,
                                                                      y_val_fold,
                                                                      model_save_path,
                                                                      in_mlflow_experiment_name,
                                                                      is_fold=True,
                                                                      val_all_folds=None,
                                                                      in_scaler=in_scaler,
                                                                      in_selector=in_selector
                                                                      )
        print('--------------------------------')
        print(f'Validation Accuracy on FOLD {fold}: {val_accuracy}')
        vals.append(val_accuracy)
        print('--------------------------------')
    mean_val = np.mean(vals)

    # Use the process_params_torch function for training and validation
    processed_params, val_accuracy, run_id = process_params_torch(params,
                                                                  x_train,
                                                                  y,
                                                                  x_train,
                                                                  y,
                                                                  model_save_path,
                                                                  in_mlflow_experiment_name,
                                                                  is_fold=False,
                                                                  val_all_folds=mean_val,
                                                                  in_scaler=in_scaler,
                                                                  in_selector=in_selector
                                                                  )
    print('--------------------------------')
    print(f'Mean validation Accuracy: {mean_val}')
    print('--------------------------------')

    return params, val_accuracy, run_id


def get_best_run_from_hyperoptim(results: List[Tuple[Dict, float, str]]) -> Tuple[Dict, float, str]:
    """
    This function retrieves the best model run from the results of the hyperparameter
    optimization.

    :param results: parameters, validation accuracy, and run ID for one run of the model

    :return: best_params: best parameters from hyperparameter optimization
             best_val_accuracy: highest validation accuracy achieved during hyperparameter
             optimization
             best_run_id: ID of the best run
    """
    # Retrieve the best model run
    best_params = {}
    best_val_accuracy = 0
    best_run_id = None
    for params, val_accuracy, run_id in results:
        if best_val_accuracy is None or val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_params = params
            best_run_id = run_id

    return best_params, best_val_accuracy, best_run_id

#
# def process_params(params, x_train, y, x_val, y_val):
#     """
#     Process a set of parameters and compute validation accuracy for TensorFlow pipeline.
#     :param params: Parameters to process.
#     :param x_train: Training data.
#     :param y: Training labels.
#     :param x_val: Validation data.
#     :param y_val: Validation labels.
#     :return: Tuple of processed parameters and validation accuracy.
#     """
#     # new_dict = restructure_dict(params)
#     print(**params)
#     # Create TensorFlow pipeline
#     pipeline_tf = Pipeline(steps=[('NN', NNTensorFlow(**params))])
#
#     # Prepare parameters for pipeline
#     params_pipeline = {pipeline_tf.steps[-1][0] + "__validation_data":
#                            (x_val.reshape(x_val.shape[0], x_val.shape[1]), y_val)}
#
#     # Fit the pipeline on training data
#     pipeline_tf.fit(x_train.reshape(x_train.shape[0], x_train.shape[1]), y,
#                     **params_pipeline)
#
#     # Retrieve the validation accuracy from the TensorFlow model
#     val_accuracy = pipeline_tf.named_steps['NN'].nn_model.history.history['val_accuracy'][-1]
#
#     # # Log the metrics and parameters in MLflow
#     # with mlflow.start_run():
#     #     mlflow.log_params(params)
#     #     mlflow.log_metric("val_accuracy", val_accuracy)
#
#     return params, val_accuracy
#
