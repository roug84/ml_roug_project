import ast

from beartype.typing import List, Dict
from sklearn.model_selection import ParameterGrid


def restructure_dict(params: dict,
                     nn_params_keys: list,
                     other_keys: list,
                     in_from_mlflow: bool
                     ) -> dict:
    """
    Restructure the parameter's dictionary. This function separates neural network-specific
    parameters from the other parameters and stores them under a new key 'nn_params'.
    :param params: Original dictionary of parameters.
    :param nn_params_keys: key to be added to nn_params
    :param other_keys: keys that remain at the same level of the original dict params
    :param in_from_mlflow: parameters coming from mlflow, should be read using ast.literal_eval
    Returns:
        dict: Restructured dictionary of parameters. The dictionary includes the original parameters
        and an added 'nn_params' key that holds a sub-dictionary of neural network-specific
        parameters.

    The neural network-specific parameters include:
        'activations'
        'in_nn'
        'input_shape'
        'output_shape'

    The other parameters remain at the top level of the dictionary:
        'batch_size'
        'cost_function'
        'learning_rate'
        'metrics'
        'n_epochs'
        'nn_key'
    """
    if in_from_mlflow:
        nn_params = {key: ast.literal_eval(params[key]) for key in nn_params_keys}

        if params['nn_key'] == 'CNN' and 'filters' in params:
            nn_params['filters'] = int(params['filters'])

        new_dict = {
            key: ast.literal_eval(params[key]) if key not in ['cost_function', 'activations',
                                                              'nn_key', 'metrics'] else params[key]
            for key in other_keys}

    else:
        nn_params = {key: params[key] for key in nn_params_keys}

        if params['nn_key'] == 'CNN' and 'filters' in params:
            nn_params['filters'] = params['filters']

        new_dict = {key: params[key] for key in other_keys}

    new_dict["nn_params"] = nn_params

    return new_dict


def flatten_dict(data: dict) -> dict:
    """
    Flatten a dictionary. This function extracts the keys from the nested dictionary
    and adds them to the outer dictionary.
    :param data: Original dictionary of parameters.
    Returns:
        dict: Flattened dictionary of parameters.
    """
    flat_dict = data.copy()  # copy the original dict
    nn_params = flat_dict.pop("nn_params")  # remove "nn_params" and store its value

    # add the keys from "nn_params" to the outer dictionary
    for key, value in nn_params.items():
        flat_dict[key] = value

    return flat_dict


def generate_param_grid_with_different_size_layers(nn_key: List[str],
                                                   input_shape: List[int],
                                                   output_shape: List[int],
                                                   batch_size: List[int],
                                                   cost_function: List[str],
                                                   learning_rate: List[float],
                                                   n_epochs: List[int],
                                                   metrics: List[str],
                                                   layer_sizes: List[List[int]],
                                                   activations: List[List[str]],
                                                   cnn_filters: List[int],
                                                   ) -> List[Dict]:
    """
    Function to generate the parameter grid for a neural network model.

    This function generates an outer parameter grid and an inner parameter grid. The outer grid
    specifies the parameters that are consistent across all models, while the inner grid specifies
    the parameters that vary between models.

    Parameters:
    :param nn_key: The type of neural network model.
    :param input_shape: Input shape of the model.
    :param output_shape: Output shape of the model.
    :param batch_size: The batch size for training the model.
    :param cost_function: The cost function for training the model.
    :param learning_rate: The learning rate for the model.
    :param n_epochs: The number of epochs for training the model.
    :param metrics: The metrics for assessing the model.
    :param layer_sizes: List of different layer sizes to be used in the grid.
    :param activations: List of different activations to be used in the grid.
    :param cnn_filters: List number of filters in CNN

    Returns:
    param_grid_inner (List[Dict[str, Union[List[int], List[str]]]]): Inner parameter grid specifying
    varying parameters. ParameterGrid(param_grid_outer) (ParameterGrid): Outer parameter grid
    specifying consistent parameters.
    """
    param_grid_outer = {
        'nn_key': nn_key,
        'input_shape': input_shape,
        'output_shape': output_shape,
        'batch_size': batch_size,
        'cost_function': cost_function,
        'learning_rate': learning_rate,
        'n_epochs': n_epochs,
        'metrics': metrics,
    }

    # Generate all combinations
    param_grid_inner = []

    # Generate all combinations
    for layers in layer_sizes:
        for activation in activations:
            for key in nn_key:
                if len(layers) == len(activation):
                    if key == 'CNN':
                        for filter_num in cnn_filters:
                            tmp_dict = {'nn_key': key, 'in_nn': layers, 'activations': activation,
                                        'filters': filter_num}
                            param_grid_inner.append(tmp_dict)
                    # elif key == 'CNN2D':
                    #     for filter_num in cnn_filters:
                    #         tmp_dict = {'nn_key': key, 'in_nn': layers,
                    #                     'activations': activation,
                    #                     'filters': filter_num}
                    #         param_grid_inner.append(tmp_dict)
                    else:
                        tmp_dict = {'nn_key': key, 'in_nn': layers, 'activations': activation}
                        param_grid_inner.append(tmp_dict)

    list_params = combine_parameters(param_grid_inner, ParameterGrid(param_grid_outer))

    return list_params


def combine_parameters(inner_params: List, outer_params: ParameterGrid) -> List[Dict]:
    """
    Combine the given inner and outer parameters into a list of dictionaries. Each dictionary
    contains a combination of an inner and outer parameter.


    :param inner_params: A list of dictionaries, each containing a set of inner parameters.
    :param outer_params: A list of dictionaries, each containing a set of outer parameters.

    :returns
    list of dict: Returns a list of dictionaries. Each dictionary is a combination of an outer and
    inner parameter dictionary.

    Example:
    inner_params = [{'a': 1, 'b': 2}, {'a': 3, 'b': 4}]
    outer_params = [{'x': 5, 'y': 6}, {'x': 7, 'y': 8}]
    combine_parameters(inner_params, outer_params)
    [{'x': 5, 'y': 6, 'a': 1, 'b': 2},
    {'x': 5, 'y': 6, 'a': 3, 'b': 4},
    {'x': 7, 'y': 8, 'a': 1, 'b': 2},
    {'x': 7, 'y': 8, 'a': 3, 'b': 4}]
    """
    all_params = []
    for outer_param in outer_params:
        for inner_param in inner_params:
            combined_param = outer_param.copy()  # copy the outer_param dict
            combined_param.update(inner_param)  # add inner parameters
            all_params.append(combined_param)
    return all_params
