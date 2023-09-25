import numpy as np
from roug_ml.utl.etl import read_data
from roug_ml.utl.etl import extract_features
import tensorflow as tf
from models.nn_models import MLPModel
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, StandardScaler
from models.pipelines.pipelines import NNTensorFlow
from models.scalers.scalers3d import NDStandardScaler
import matplotlib.pyplot as plt
np.random.seed(1)
tf.random.set_seed(2)
import os
import scipy.io as sio


if __name__ == '__main__':
    # file_name = 'sp623_pat1'
    main_data_path = 'C:/Users/kenso/Documents/MATLAB/data_from_cea'
    fol, _path, list_of_patients = next(os.walk((main_data_path)))

    for patient_file in list_of_patients:
        print(patient_file[:-4])
        if patient_file[:-4] in ['sp623_pat_24', 'sp623_pat_27']:
            continue
        path_to_bd1 = os.path.join(main_data_path, patient_file)
        list_of_pat = ['patient' + str(i + 1) for i in range(20)]
        dataset = {}
        mat_contents = sio.loadmat(path_to_bd1)
        # try:
        dataset['clinical_study'] = patient_file[:3]
        try:
            dataset['name'] = mat_contents['data_patient_i']['patientInfo'][0][0]['name'][0][0][0]
        except:
            dataset['name'] ='not name'

        # in_dataset['sex'] = mat_contents['data_patient_i']['patientInfo'][0][0]['sex'][0][0][0]
        try:
            dataset['weight'] = float(mat_contents['data_patient_i']['patientInfo'][0][0]['weight'][0][0][0])
        except:
            dataset['weight'] = 0
        dataset['age'] = int(mat_contents['data_patient_i']['patientInfo'][0][0]['age'][0][0][0])
        # in_dataset['height'] = float(mat_contents['data_patient_i']['patientInfo'][0][0]['height'][0][0][0])
        dataset['cgm'] = mat_contents['data_patient_i']['glycemia'][0, 0]['CGM1'][0, 0]['value'][0, 0]
        dataset['bolus'] = mat_contents['data_patient_i']['insulin'][0, 0]['bolus'][0, 0]['value'][0, 0]
        dataset['basal'] = mat_contents['data_patient_i']['insulin'][0, 0]['basal'][0, 0]['value'][0, 0]
        dataset['meal'] = mat_contents['data_patient_i']['meal'][0, 0]['value'][0, 0]
        dataset['AGcount_m'] = mat_contents['data_patient_i']['sport'][0, 0]['AGcount_m'][0][0]
        dataset['activity_type'] = mat_contents['data_patient_i']['sport'][0, 0]['activity_type'][0, 0]
        dataset['EE_wmlm'] = mat_contents['data_patient_i']['sport'][0, 0]['EE_wmlm'][0, 0]
        dataset['BPM'] = mat_contents['data_patient_i']['sport'][0, 0]['BPM'][0, 0]
        dataset['intensity'] = mat_contents['data_patient_i']['sport'][0, 0]['value'][0, 0]

        # in_dataset = pd.read_csv(os.path.join(path_to_bd1, pat + 'gly.csv'), names=colnames, header=None)

        dataset['CGM'] = dataset['cgm'] * 18
        fig, axes = plt.subplots(nrows=5, ncols=2, sharex=True)
        axes[0, 0].set_title(dataset['clinical_study'] + '_' + dataset['name'])
        axes[0, 0].plot(dataset['CGM'], 'ro')
        axes[0, 0].set_ylabel("CGM")

        axes[3, 0].plot(dataset['basal'], 'bo')
        axes[3, 0].set_ylabel("basal")

        axes[2, 0].plot(dataset['bolus'], 'bo')
        axes[2, 0].set_ylabel("bolus")

        axes[1, 0].plot(dataset['meal'], 'bo')
        axes[1, 0].set_ylabel("meal")

        axes[0, 1].plot(dataset['EE_wmlm'], 'bo')
        axes[0, 1].set_ylabel("EE_wmlm")

        axes[3, 1].plot(dataset['intensity'], 'bo')
        axes[3, 1].set_ylabel("sport")

        axes[1, 1].plot(dataset['AGcount_m'], 'bo')
        axes[2, 1].set_ylabel("AGcount_m")

        axes[3, 1].plot(dataset['BPM'], 'bo')
        axes[3, 1].set_ylabel("BPM")

        axes[4, 1].plot(dataset['activity_type'], 'bo')
        axes[4, 1].set_ylabel("activity_type")
        # plt.title(pat)
        plt.savefig(os.path.join(main_data_path, 'figs', dataset['clinical_study'] + '_' + dataset['name'] + '.png'))
        plt.close()
        # except:
        #     continue
        # plt.show()