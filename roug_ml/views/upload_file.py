

import os
import pandas as pd
import numpy as np
import json
# import matplotlib.pyplot as plt
import tensorflow as tf
from flask import Blueprint, render_template, redirect, Response, request, url_for, flash
# from app_home_sensors import app
# app_home_sensors
from werkzeug.utils import secure_filename
# from flask_mail import Mail, Message
from etl.extract_utl import extract_features
from models.pipelines.pipelines import NNTensorFlow
import requests
activities_label = {0: "L0: nothing", 1: "L1: Standing still (1 min)", 2: "L2: Sitting and relaxing (1 min)",
                        3: "L3: Lying down (1 min)", 4: "L4: Walking (1 min)",
                        5: "L5: Climbing stairs (1 min)", 6: "L6: Waist bends forward (20x)",
                        7: "L7: Frontal elevation of arms (20x)",
                        8: "L8: Knees bending (crouching) (20x)", 9: "L9: Cycling (1 min)", 10: "L10: Jogging (1 min)",
                        11: "L11: Running (1 min)", 12: "L12: Jump front & back (20x)"}
ALLOWED_EXTENSIONS = {'csv', 'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif', 'log'}
UPLOAD_FOLDER = '/Users/hector/DiaHecDev/data'

bp10 = Blueprint('bp10', __name__, template_folder='templates')
params2 = {
        'nn_key': 'MLP',
        'nn_params': {
            'input_shape': 126,
            'output_shape': 13,
            'in_nn': [200, 300],
            'activations': ['relu', 'relu']
        },
        'batch_size': 10,
        'cost_function': 'categorical_crossentropy',
        'learning_rate': 0.001,
        'n_epochs': 36,
        'metrics': ['accuracy']}


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@bp10.route('/upload_file', methods=['POST', 'GET'])
def view():
    if request.method == 'POST':
        print("post")
        print(request)
        print(request.files)
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        print("file in request.files")
        file = request.files['file']
        input_df = pd.read_csv(file, sep="\t", header=None)
        # input_df[24] = file[8:-4]
        print(input_df.head())
        x = np.asarray(input_df[[0, 1, 2]])
        model = NNTensorFlow(**params2)
        model.load('/Users/hector/DiaHecDev/results/Models_DL', 'servingpa')
        sample_freq = 50  # Hz
        sampling_rate = 1 / sample_freq
        windo_size = 1  # [s]
        points_for_mean = int(windo_size / sampling_rate)
        batch_input = []
        mean_ps_list, mean_ps_test_list, median_ps_list, median_ps_test_list = ([] for i in range(4))
        for i in range(0, len(x), 100):
            y_list = []
            x_list = []
            x_var_list = []
            for j in range(i, 511 + i, int(points_for_mean / 2)):
                x_tmp_train = x[j:j + points_for_mean, :]
                # Compute mean
                mean = np.mean(x_tmp_train, axis=0)
                # Compute variance
                vari = np.var(x_tmp_train, axis=0)
                # Compute mean frequency of power spectrum
                ps_train = np.abs(np.fft.fft(x_tmp_train)) ** 2
                mean_ps_train = np.mean(ps_train, axis=0)
                median_ps_train = np.median(ps_train, axis=0)
                x_list.append(mean)
                x_var_list.append(vari)


            x_act_train = np.asarray(x_list)
            x_act_var = np.asarray(x_var_list)
            # for j in np.arange(0, 511, 511):
            x_act_train = np.concatenate((x_act_train, x_act_var), axis=0)
            x_train = x_act_train.reshape(1, x_act_train.shape[0] * x_act_train.shape[1])
            batch_input.append(x_train)
            # print(tf.argmax(model.predict(x_train), axis=1))        # feat_y_list.append(io_final_data_set['y'][i])
        x_batch = np.asarray(batch_input).reshape(len(batch_input), 126)

        MODEL_URI = 'http://localhost:8501/v1/models/servingpa:predict'
        acts = []
        for i in range(0, 200*int(len(x_batch)/200), 200):
            data = json.dumps({
                'inputs': x_batch[i:i + 200, :].reshape(1, 200, 126).tolist()
            })
            response = requests.post(MODEL_URI, data=data)
            result = json.loads(response.text)
            # print(result)
            prediction = np.squeeze(result['outputs'][0])
            # print(prediction)
            acts.append(tf.argmax(prediction, axis=1).numpy())
        list_acts = []
        for j in range(len(np.hstack(acts))):
            list_acts.append(activities_label[np.hstack(acts)[j]])  # feat_y_list.append(io_final_data_set['y'][i])
        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            print("file empty")
            flash('No selected file')
            return redirect(request.url)

        # print("file not empty")
        # print("file", file)
        # print("allowed_file", allowed_file(file.filename))
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(UPLOAD_FOLDER, filename))
            # send_test_mail()
            return render_template('temperature.html', notes=list_acts)#['-'.join(list_acts)])
    return render_template('upload_file.html')