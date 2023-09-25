import numpy as np
from roug_ml.utl.etl import read_data
from roug_ml.utl.etl import integer_to_onehot, extract_data_for_train_test
from roug_ml.utl.etl import extract_features
import tensorflow as tf
from models.pipelines.pipelines import NNTensorFlow
from evaluation.eval_utl import compute_accuracy_from_soft_max

from models.pipelines.pipeline_utl import PipeModel
np.random.seed(1)
tf.random.set_seed(2)


if __name__ == '__main__':
    path_to_bd1 = '/Users/hector/DiaHecDev/data/MHEALTHDATASET'
    dataset = read_data(path_to_bd1)

    ynum = np.asarray(dataset[23])  # 23 i the type of activity
    # print(ynum)
    y_onehot = integer_to_onehot(ynum)
    x_chest = np.asarray(dataset[[0, 1, 2]])  # chest
    x_ankle = np.asarray(dataset[[5, 6, 7]])  # left-ankle
    x_rightarm = np.asarray(dataset[[14, 15, 16]])  # right-lower-arm
    user = np.asarray(dataset[24])
    dataset_dict = {'x_chest': x_chest, 'x_ankle': x_ankle, 'x_rightarm': x_rightarm, 'yhot': y_onehot, 'y_num': ynum,
                    'User': user}
    # Create labels
    activities_label = {0: "L0: nothing", 1: "L1: Standing still (1 min)", 2: "L2: Sitting and relaxing (1 min)",
                        3: "L3: Lying down (1 min)", 4: "L4: Walking (1 min)",
                        5: "L5: Climbing stairs (1 min)", 6: "L6: Waist bends forward (20x)",
                        7: "L7: Frontal elevation of arms (20x)",
                        8: "L8: Knees bending (crouching) (20x)", 9: "L9: Cycling (1 min)", 10: "L10: Jogging (1 min)",
                        11: "L11: Running (1 min)", 12: "L12: Jump front & back (20x)"}

    # Create in_dataset
    final_data_set = extract_data_for_train_test(in_num_of_samples=10,
                                                 in_dataset_dict=dataset_dict,
                                                 in_activities_label=activities_label,
                                                 in_points_in_each_set=511)

    final_data_set = extract_features(final_data_set, in_list_positions=['rightarm', 'chest', 'ankle'])

    #
    x_train = np.asarray(final_data_set['x_ankle_train'])
    y_train_oh = np.asarray(final_data_set['y'])
    x_test = np.asarray(final_data_set['x_ankle_test'])
    y_test_oh = np.asarray(final_data_set['y'])

    params2 = {
        'nn_key': 'MLP',
        'nn_params': {
            'input_shape': final_data_set['x_feat'][0].shape[1],
            'output_shape': 13,
            'in_nn': [400, 500],
            'activations': ['tanh', 'relu']
        },
        'batch_size': 10,
        'cost_function': 'categorical_crossentropy',
        'learning_rate': 0.0001,
        'n_epochs': 36,
        'metrics': ['accuracy']}

    # Create pipeline
    # model = Pipeline(steps=[('RobustScaler', StandardScaler()), ('NN', NNTensorFlow(**params2))])
    pipe = PipeModel(steps=[('NN', NNTensorFlow(**params2))])

    X_val = np.asarray(final_data_set['x_feat_test'])
    X = np.asarray(final_data_set['x_feat'])
    y = y_train_oh.reshape(130, 13)
    params_pipeline = {}
    params_pipeline[pipe.steps[-1][0] + "__validation_data"] = (X_val.reshape(X_val.shape[0], X_val.shape[2]), y)
    pipe.fit(X.reshape(X.shape[0], X.shape[2]), y, **params_pipeline)

    path_model = '/Users/hector/DiaHecDev/data/model'
    model_name = "models_3ax2"

    #pipe.save(model_path=path_model, model_name=model_name)

    pipe.load(model_path=path_model, model_name=model_name)
    y_pred = pipe.predict(X_val.reshape(X_val.shape[0], X_val.shape[2]))
    acc = compute_accuracy_from_soft_max(y=y,y_pred=y_pred)
    print(acc)
    # model.evaluate(np.asarray(final_data_set['x_feat_test']), y_train_oh.reshape(130, 1, 13))


