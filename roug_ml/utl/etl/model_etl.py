"""
Util functions to save and load models.
To be used by methods save and load of transforms and models inherinting from
templates.
Eventually, to replace `deep_learn_utl.py`.
"""
import logging
log = logging.getLogger(__name__)  # noqa: E402

import os
import joblib

# import tensorflow as tf
import pandas as pd

from roug_ml.utl.paths_utl import create_dir_path


def save_default_model(model, model_path, model_name):
    """Save standard model (sklearn like).
    :param model: The model to save.
    :type model: any model instance
    :param model_path: The path where to save it.
    :type model_path: string
    :param model_name: The name of the model.
    :type model_name: string
    """
    path = os.path.join(model_path, model_name + '.gz')
    log.debug('dumping model in: %s', path)

    joblib.dump(model, path)


def save_default_model_as_csv(model, model_path, model_name):
    """Save standard model params (sklearn like) as csv.
    :param model: The model to save.
    :type model: any model instance
    :param model_path: The path where to save it.
    :type model_path: string
    :param model_name: The name of the model.
    :type model_name: string
    """

    # only to export RobustScaler
    if model_name == 'robust_scaler':
        path = os.path.join(model_path, model_name + '.csv')

        data_scaler = pd.DataFrame()
        data_scaler['center_'] = model.center_
        data_scaler['scale_'] = model.scale_
        data_scaler.to_csv(path)


def load_default_model(model_path, model_name):
    """Load standard model (sklearn like).
    :param model_path: The path where to save it.
    :type model_path: string
    :param model_name: The name of the model.
    :type model_name: string
    :return: The loaded model.
    :rtype: any model instance
    """
    path = os.path.join(model_path, model_name + '.gz')
    log.debug('load model from: %s', path)

    return joblib.load(path)


def save_keras_model_old(model, model_path, model_name="model"):
    """Save keras model in two separate files.
    :param model: The keras model to save.
    :type model: tf.Keras.Model
    :param model_path: The path where to save it.
    :type model_path: string
    :param model_name: The name of the model.
    :type model_name: string
    """
    path_json = os.path.join(model_path, model_name + '_archi.json')
    path_h5 = os.path.join(model_path, model_name + '_weights.h5')
    log.debug('dumping model in: %s - %s', path_json, path_h5)
    model_json = model.to_json()
    with open(path_json, "w") as write_file:
        write_file.write(model_json)

    model.save_weights(path_h5)


def load_keras_model_old(model_path, model_name):
    """Load keras model from two separate files.
    :param model_path: The path where to save it.
    :type model_path: string
    :param model_name: The name of the model.
    :type model_name: string
    :return: The keras model loaded.
    :rtype: tf.Keras.Model
    """
    path_json = os.path.join(model_path, model_name + '_archi.json')
    path_h5 = os.path.join(model_path, model_name + '_weights.h5')
    log.debug('load model from: %s - %s', path_json, path_h5)

    json_file = open(path_json, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = tf.keras.models.model_from_json(loaded_model_json)
    model.load_weights(path_h5)

    return model


def save_keras_model(model, model_path, model_name):
    """Save keras model in two separate files.
    :param model: The keras model to save.
    :type model: tf.Keras.Model
    :param model_path: The path where to save it.
    :type model_path: string
    :param model_name: The name of the model.
    :type model_name: string
    """
    create_dir_path(os.path.join(model_path, model_name))
    model.save(os.path.join(model_path, model_name))


def save_keras_model_as_tflite(model, model_path, model_name):
    """Save keras model as a TensorFlow Lite model.
    :param model: The keras model to save.
    :type model: tf.Keras.Model
    :param model_path: The path where to save it.
    :type model_path: string
    :param model_name: The name of the model.
    :type model_name: string
    """
    # Convert the model.
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()

    # Save the model
    with open(os.path.join(model_path, model_name, 'saved_model.tflite'), 'wb') as f:
        f.write(tflite_model)


def load_keras_model(model_path, model_name, custom_loss=None):
    """Load keras model from two separate files.
    :param model_path: The path where to save it.
    :type model_path: string
    :param model_name: The name of the model.
    :type model_name: string
    :param custom_loss: customized loss function
    :type custom_loss: function
    :return: The keras model loaded.
    :rtype: tf.Keras.Model
    """
    model = tf.keras.models.load_model(os.path.join(model_path, model_name),
                                       custom_objects={'loss': custom_loss})
    return model
