import torch
from torchvision import datasets, transforms
import os
from sklearn.pipeline import Pipeline
# Get current working directory
import sys
import mlflow
import numpy as np
from roug_ml.utl.etl.transforms_utl import integer_to_onehot
from roug_ml.utl.data_vizualization.images_vizualization import visualize_incorrect_classifications

current_path = os.getcwd()
sys.path.append(os.path.join(current_path, '../..'))

from roug_ml.bases import MLPipeline
from roug_ml.configs.my_paths import RESULTS_PATH
from roug_ml.utl.paths_utl import create_dir
from roug_ml.utl.dowload_utils import download_kaggle_dataset
from roug_ml.utl.file_operations import move_files_to_labels_dir, extract_zip
from roug_ml.utl.data_vizualization.images_vizualization import imshow
from roug_ml.utl.dataset_split import load_transform_image_data_from_path, split_image_data
from roug_ml.models.pipelines.pipelines import NNTorch
from roug_ml.utl.evaluation.eval_utl import calc_loss_acc
from roug_ml.utl.evaluation.multiclass import compute_multiclass_confusion_matrix
from roug_ml.models.hyperoptimization import parallele_hyper_optim
from roug_ml.utl.mlflow_utils import get_or_create_experiment
from roug_ml.models.hyperoptimization import get_best_run_from_hyperoptim
from roug_ml.utl.mlflow_utils import get_best_run
from roug_ml.utl.parameter_utils import restructure_dict
from torch.utils.data import Subset
from torchvision.transforms import transforms
from typing import Tuple


def extract_data_from_subset(subset: Subset) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extracts data (images) and labels from a Subset.

    :param subset: Subset of data.
    :return: Numpy array of images and numpy array of labels.
    """
    X, y = [], []

    for img, label in subset:
        # Convert PyTorch tensor to a numpy array and add it to the list.
        X.append(img.numpy())
        y.append(label)

    # Convert lists to numpy arrays.
    X = np.stack(X)  # This assembles the separate arrays in the list into one array.
    y = np.array(y)

    return X, y


class CatsDogsPipeline(MLPipeline):
    """
    A pipeline to handle preprocessing, training, and evaluating a model for the Cats vs Dogs
    classification task.
    """
    def __init__(self, in_params, in_res_path, in_train_path, in_mlflow_experiment_name):
        """
        Initializes the pipeline with the given parameters.
        :param in_params: A dictionary containing parameters for the neural network model.
        :param in_res_path: Path to save the results.
        :param in_train_path: Path to the training data.
        """

        super().__init__(in_mlflow_experiment_name)
        self.mlflow_experiment_name = in_mlflow_experiment_name
        self.test_loader = None
        self.val_loader = None
        self.train_loader = None
        self.test_data = None
        self.val_data = None
        self.train_data = None
        self.transform = None
        self.params = in_params
        self.nn_params = in_params['nn_params']
        self.nn_params_keys = list(self.nn_params.keys())
        self.other_keys = list(self.params.keys())
        self.other_keys.remove('nn_params')

        self.nn_key = in_params['nn_key']
        self.cost_function = in_params['cost_function']
        self.batch_size = in_params['batch_size']
        self.learning_rate = in_params['learning_rate']
        self.n_epochs = in_params['n_epochs']
        self.metrics = in_params['metrics']

        self.pipeline_torch = Pipeline(steps=[('NN', NNTorch(**in_params))])

        self.train_path = in_train_path
        self.res_path = in_res_path

        self.set_mlflow_params()

        self.num_workers = 7
        self.framework = 'torch'
        self.results_path = ''
        self.re_optimize = False
        
    def set_mlflow_params(self):
        """
        Sets the tracking URI for mlflow and initializes the mlflow experiment
        """
        mlflow.set_tracking_uri("http://localhost:8000")
        self.mlflow_experiment_id = get_or_create_experiment(self.mlflow_experiment_name)

    def run(self):
        """
        Run the entire pipeline from preprocessing to prediction and evaluation.
        """
        self.preprocess()
        best_params, best_run_id = self.fit()
        # Load the best model
        self.pipeline_torch = mlflow.sklearn.load_model("runs:/{}/pipeline".format(best_run_id))

        X_test, y_test = extract_data_from_subset(self.test_data)
        y_test_hot = integer_to_onehot(y_test)
        predictions = self.pipeline_torch.predict(X_test)

        # Convert predictions to tensor if it's numpy array
        if isinstance(predictions, np.ndarray):
            predictions = torch.from_numpy(predictions)

        if isinstance(y_test_hot, np.ndarray):
            y_test_hot = torch.from_numpy(y_test_hot)

        _, val_acc = calc_loss_acc(predictions, y_test_hot, None)
        print(val_acc)

        compute_multiclass_confusion_matrix(targets=y_test_hot.numpy(),
                                            outputs_list=[predictions.numpy()],
                                            model_names=['NN'],
                                            class_labels=None)

        # Call the function
        visualize_incorrect_classifications(self.pipeline_torch, X_test, y_test)

        # test_predictions = self.predict(self.test_loader)  # Pass in test loader
        # self.evaluate(self.test_loader)  # Same here

    def preprocess(self):
        """
        Preprocess the data by applying transformations and splitting the data into train, validation, and test sets.
        """
        # Define transformations
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),  # resize images to 224x224
            transforms.ToTensor(),  # convert image to PyTorch Tensor data type
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            # normalize images
        ])

        # Load datasets
        # self.train_data, self.val_data, self.test_data = \
        #     split_image_data_from_path(main_path=self.train_path, train_ratio=0.7,
        #                                val_ratio=0.15,
        #                                test_ratio=0.15,
        #                                transform=self.transform)

        all_data = load_transform_image_data_from_path(main_path=self.train_path,
                                                       transform=self.transform)
        self.train_data, tmp_data = split_image_data(data=all_data,
                                                     train_ratio=0.7,
                                                     val_ratio=0.3
                                                     )
        self.val_data, self.test_data = split_image_data(data=tmp_data,
                                                         train_ratio=0.5,
                                                         val_ratio=0.5
                                                         )

        # def split_image_data_from_path(main_path: str,
        #                                train_ratio: float,
        #                                val_ratio: float,
        #                                test_ratio: float,
        #                                transform: transforms) -> Tuple[Dataset, Dataset, Dataset]:
        #     """
        #     Split data into training, validation, and testing sets.
        #
        #     :param main_path: The path to the main directory containing all the images.
        #     :param train_ratio: The ratio of data to be used for training.
        #     :param val_ratio: The ratio of data to be used for validation.
        #     :param test_ratio: The ratio of data to be used for testing.
        #     :param transform:
        #
        #     :return: Training dataset, validation dataset, and testing dataset.
        #     """
        #
        #     # Load all data
        #     all_data = ImageFolder(main_path, transform=transform)
        #
        #     # Ensure ratios sum to 1
        #     assert train_ratio + val_ratio + test_ratio == 1, "Ratios must sum to 1"
        #
        #     # Set the split sizes
        #     train_size = int(train_ratio * len(all_data))
        #     val_size = int(val_ratio * len(all_data))
        #     test_size = len(all_data) - train_size - val_size  # remaining for testing
        #
        #     # Split the data
        #     train_data, val_data, test_data = random_split(all_data, [train_size, val_size, test_size])
        #
        #     return train_data, val_data, test_data

        # # Create dataloaders
        # self.train_loader = torch.utils.data.DataLoader(self.train_data,
        #                                                 batch_size=self.batch_size,
        #                                                 shuffle=True,
        #                                                 num_workers=4)
        # self.val_loader = torch.utils.data.DataLoader(self.val_data,
        #                                               batch_size=self.batch_size,
        #                                               shuffle=True,
        #                                               num_workers=4)
        #
        # self.test_loader = torch.utils.data.DataLoader(self.test_data,
        #                                                batch_size=self.batch_size,
        #                                                shuffle=True,
        #                                                num_workers=4)

    def fit(self):
        """
        Fits the model to the training data.
        """
        if self.re_optimize:
            # Optimize and get best run

            input_shape = (3, 224, 224)
            output_shape = 2

            nn_params_2 = {
                'input_shape': input_shape,  # Input shape (Number of Channels, Height, Width)
                'output_shape': output_shape,  # Number of output nodes
                'filters': (32, 64),  # Number of filters in the convolutional layers
                'fc_nodes': 1000,  # Number of nodes in the fully connected layers
            }

            params_cnn2d2 = {'nn_params': nn_params_2, 'nn_key': 'CNN2D',
                             "cost_function": torch.nn.CrossEntropyLoss(),
                             'batch_size': 128, 'learning_rate': 0.01, 'n_epochs': 5,
                             'metrics': 'accuracy'}

            params_cnn2d3 = {'nn_params': nn_params_2, 'nn_key': 'CNN2D',
                             "cost_function": torch.nn.CrossEntropyLoss(),
                             'batch_size': 128, 'learning_rate': 0.001, 'n_epochs': 5,
                             'metrics': 'accuracy'}

            nn_params_3 = {
                'input_shape': input_shape,  # Input shape (Number of Channels, Height, Width)
                'output_shape': output_shape,  # Number of output nodes
                'filters': (64, 128),  # Number of filters in the convolutional layers
                'fc_nodes': 50,  # Number of nodes in the fully connected layers
            }

            params_cnn2d4 = {'nn_params': nn_params_3, 'nn_key': 'CNN2D',
                             "cost_function": torch.nn.CrossEntropyLoss(),
                             'batch_size': 128, 'learning_rate': 0.001, 'n_epochs': 5,
                             'metrics': 'accuracy'}

            list_params = [self.params,
                           # params_cnn2d2, params_cnn2d3, params_cnn2d4
                           ]

            x_train, y_train = extract_data_from_subset(self.train_data)
            x_val, y_val = extract_data_from_subset(self.val_data)

            # plot_label_distribution_from_arrays(label_mapping={0: '0', 1: '1'},
            #                                     Train=(x_train, y_train),
            #                                     Validation=(x_val, y_val)
            #                                     )

            y_train = integer_to_onehot(y_train)  # , n_labels=params['nn_params']['output_shape'])
            y_val = integer_to_onehot(y_val)

            results = parallele_hyper_optim(in_num_workers=self.num_workers,
                                            x_train=x_train,
                                            y=y_train,
                                            x_val=x_val,
                                            y_val=y_val,
                                            param_grid_outer=list_params,
                                            in_framework=self.framework,
                                            model_save_path=self.results_path,
                                            in_mlflow_experiment_id=self.mlflow_experiment_id
                                            )

            params_pipeline = {
                self.pipeline_torch.steps[-1][0] + "__validation_data": (x_val, y_val)}

            self.pipeline_torch.fit(x_train, y_train, **params_pipeline)
            best_params, best_val_accuracy, best_run_id = get_best_run_from_hyperoptim(results)
        else:
            # Get the best run from mlopt
            best_run_id, best_params = get_best_run(self.mlflow_experiment_name, "val_accuracy")
            best_params = restructure_dict(best_params, self.nn_params_keys, self.other_keys,
                                           in_from_mlflow=True)
        return best_params, best_run_id

    def load_data_set(self):
        """
        Loads the dataset from Kaggle and extracts the zip files.
        """
        download_kaggle_dataset(dataset='dogs-vs-cats',
                                username="kensou",
                                key="7aefffc32036aa470ac86d76a4a80576",
                                download_path=self.res_path)
        extract_zip(os.path.join(self.res_path, 'dogs-vs-cats.zip'), self.res_path)
        extract_zip(os.path.join(self.res_path, 'test1.zip'), self.res_path)
        extract_zip(os.path.join(self.res_path, 'train.zip'), self.res_path)

        move_files_to_labels_dir(os.path.join(self.res_path, 'train'), ['cat', 'dog'])

    def displays_samples(self):
        """
        Display samples of the images at various stages of the preprocessing pipeline.
        """
        # Define transformations
        # resize images to 224x224
        resize_transform = transforms.Resize((224, 224))
        # convert image to PyTorch Tensor data type
        to_tensor_transform = transforms.ToTensor()
        # normalize images
        normalize_transform = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                   std=[0.229, 0.224, 0.225])

        # Load datasets
        train_data = datasets.ImageFolder(self.train_path)

        # Get label names
        label_names = train_data.classes

        # Select an image
        img, label = train_data[0]

        # Show original image
        imshow(img, f'{label_names[label]} - Original')

        # Apply resize
        img = resize_transform(img)
        imshow(img, f'{label_names[label]} - Resize')

        # Convert to tensor
        img = to_tensor_transform(img)
        imshow(img, f'{label_names[label]} - To Tensor')

        # Normalize
        img = normalize_transform(img)
        imshow(img, f'{label_names[label]} - Normalize')

    def predict(self, test_loader):
        """
        Makes predictions using the trained model.
        """
        self.pipeline_torch.eval()
        all_preds = []
        for data, _ in test_loader:
            output = self.pipeline_torch.predict(data)
            _, predicted = torch.max(output.data, 1)
            all_preds.extend(predicted.numpy())
        return all_preds

    def evaluate(self, test_loader):
        """
        Evaluates the model on the test data.
        """
        correct = 0
        total = 0
        with torch.no_grad():
            for data, labels in test_loader:
                outputs = self.pipeline_torch.predict(data)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        print(f"Test Accuracy: {100 * correct / total}%")


if __name__ == '__main__':
    res_path = os.path.join(RESULTS_PATH, 'image_example')
    create_dir(res_path)

    train_path = os.path.join(res_path, 'train')
    # Assume the input image shape is (3, 128, 128) and output classes are 2
    # Replace with actual values
    input_shape = (3, 224, 224)
    output_shape = 2

    nn_params_2 = {
        'input_shape': input_shape,  # Input shape (Number of Channels, Height, Width)
        'output_shape': output_shape,  # Number of output nodes
        'filters': (32, 64),  # Number of filters in the convolutional layers
        'fc_nodes': 1000,  # Number of nodes in the fully connected layers
    }

    params_cnn2d = {'nn_params': nn_params_2, 'nn_key': 'CNN2D',
                    "cost_function": torch.nn.CrossEntropyLoss(),
                    'batch_size': 64, 'learning_rate': 0.0001, 'n_epochs': 5, 'metrics': 'accuracy'}
    # FlexCNN2D
    nn_params = {
        'input_shape': input_shape,  # Input shape (Number of Channels, Height, Width)
        'output_shape': output_shape,  # Number of output nodes
        'filters': (32, 64),  # Number of filters in the convolutional layers
        'fc_nodes': (1024, 512),  # Number of nodes in the fully connected layers
        'activation': 'relu'  # Activation function to be used
    }

    params_flex = {'nn_params': nn_params, 'nn_key': 'FlexCNN2D',
                   "cost_function": torch.nn.CrossEntropyLoss(),
                   'batch_size': 64, 'learning_rate': 0.0001, 'n_epochs': 100, 'metrics': 'accuracy'}

    nn_params_alex = {
        'output_shape': output_shape,  # Number of output nodes
    }

    #  AlexNet
    params_alexnet = {'nn_params': nn_params_alex, 'nn_key': 'AlexNet',
                      "cost_function": torch.nn.CrossEntropyLoss(),
                      'batch_size': 64, 'learning_rate': 0.001, 'n_epochs': 2,
                      'metrics': 'accuracy'}

    conv_module_params = [
        {
            "in_channels": 3, "out_channels": 64, "kernel_size": 11, "stride": 4, "padding": 2,
            "pool_kernel_size": 3, "pool_stride": 2
        },
        {
            "in_channels": 64, "out_channels": 192, "kernel_size": 5, "stride": 1, "padding": 2,
            "pool_kernel_size": 3, "pool_stride": 2
        },
        {
            "in_channels": 192, "out_channels": 384, "kernel_size": 3, "stride": 1, "padding": 1,
            "pool_kernel_size": 1, "pool_stride": 1
        },
        {
            "in_channels": 384, "out_channels": 256, "kernel_size": 3, "stride": 1, "padding": 1,
            "pool_kernel_size": 1, "pool_stride": 1
        },
        {
            "in_channels": 256, "out_channels": 256, "kernel_size": 3, "stride": 1, "padding": 1,
            "pool_kernel_size": 3, "pool_stride": 2
        }

    ]

    classifier_params = [
        {"in_features": 9216, "out_features": 4096},
        {"in_features": 4096, "out_features": 4096},
        {"in_features": 4096, "out_features": 1000},
    ]

    avgpool_output_size = (6, 6)  # This should match the size after your conv layers

    nn_params_power_flex = {
        'conv_module_params': conv_module_params,
        'classifier_params': classifier_params,
        'avgpool_output_size': avgpool_output_size,
        'output_shape': output_shape  # Number of output nodes
    }

    params_power_flex = {'nn_params': nn_params_power_flex, 'nn_key': 'FlexibleCNN2DTorchPower',
                         "cost_function": torch.nn.CrossEntropyLoss(),
                         'batch_size': 64, 'learning_rate': 0.001, 'n_epochs': 50,
                         'metrics': 'accuracy'}

    pipeline = CatsDogsPipeline(params_alexnet, res_path, train_path,
                                in_mlflow_experiment_name="cats_and_dogs")
    pipeline.run()
