import torch
import torch.optim as optim
import torchvision
from torch.utils.data import TensorDataset, DataLoader, RandomSampler

import json
from sklearn.metrics import accuracy_score
import os
import numpy as np
import random

from roug_ml.utl.evaluation.eval_utl import calc_loss_acc
from roug_ml.models.nn_models import MLPTorchModel, CNNTorch, FlexibleCNN2DTorch, \
    CNN2DTorch, FlexibleCNN2DTorchPower, MyTorchModel#, MLPModel


# import tensorflow as tf
from roug_ml.utl.etl.model_etl import save_keras_model, load_keras_model
from roug_ml.utl.set_seed import set_seed

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

# For PyTorch
torch.manual_seed(42)

# Also set the seed for the CUDA RNG if you're using a GPU
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)
    torch.cuda.manual_seed_all(42)

# Additional options for deterministic operations
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


class NNTensorFlow():
    """Model to wrap around tensorflow neural networks.
    """

    def __init__(self,
                 nn_key='MLP',
                 nn_params={},
                 n_epochs=10,
                 batch_size=100,
                 learning_rate=0.001,
                 cost_function="mean_squared_error",
                 metrics=['accuracy'],
                 verbose=True):
        """Instantiate the model.
        :param nn_params: Parameters to pass to the neural network.
        :type nn_params: dict
        :param n_epochs: The number of epochs.
        :type n_epochs: int
        :param batch_size: The batch size.
        :type batch_size: int
        :param learning_rate: The learning rate of the optimizer.
        :type learning_rate: float
        :param cost_function: Identifier of the cost function. Available losses in keras are described here:
        https://keras.io/api/losses/#available-losses
        :type cost_function: str
        :param verbose: Verbose to pass to model fit step.
        :type verbose: boolean
        """
        super().__init__()
        self.nn_params = nn_params
        if nn_key == 'MLP':
            self.nn_model = MLPModel(**self.nn_params)
        else:
            self.nn_model = MLPModel(**self.nn_params) #AEModel(**nn_params)
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.cost_function = cost_function
        self.verbose = verbose
        self.metrics = metrics

    def fit(self, X, y, validation_data=None, continue_training=False):
        """Train the neural network.
        :param X: Training data.
        :type X: array-like or sparse matrix of shape (n_samples, n_features)
        :param y: Target values.
        :type y: array-like of shape (n_samples,) or (n_samples, n_targets)
        :param validation_data: (X, y) validation data
        :type validation_data: X is array-like or sparse matrix of shape (n_samples_val, n_features) and y is
                               Target validation values.
        :param continue_training: True to not re-instantiate the model each it is trained
        :type continue_training: bool
        :return: An instance of self.
        """

        if not continue_training:
            # Re-instantiate neural network in case of hyper-optimization
            self.nn_model = MLPModel(**self.nn_params)

        # Set optimizer
        optimizer_choice = tf.keras.optimizers.Adam(
            lr=self.learning_rate,
            beta_1=0.9,
            beta_2=0.999,
            amsgrad=False)

        # Compile
        # if self.cost_function in COST_FUNCTIONS:
        #     self.cost_function = COST_FUNCTIONS[self.cost_function]

        self.nn_model.compile(
            loss=self.cost_function,
            optimizer=optimizer_choice,
            metrics=self.metrics
        )

        # Train the model
        self.nn_model.fit(
            X, y,
            validation_data=validation_data,
            batch_size=self.batch_size,
            epochs=self.n_epochs,
            verbose=self.verbose)

        return self

    def predict(self, X):
        """Predict using the neural network.
        :param X: Samples.
        :type X: array_like or sparse matrix, shape (n_samples, n_features)
        :return: Predicted values.
        """
        return self.nn_model.predict(X)

    def save(self, model_path, model_name):
        """Saving the model.
        :param model_path: The path where to save it.
        :type model_path: string
        :param model_name: The name of the model.
        :type model_name: string
        """
        # pass
        save_keras_model(self.nn_model, model_path, model_name)
        # save_keras_model_as_tflite(self.nn_model, model_path, model_name)
        #
        # save_json(
        #     {
        #         "n_epochs": self.n_epochs,
        #         "batch_size": self.batch_size,
        #         "learning_rate": self.learning_rate,
        #         "cost_function": str(self.cost_function),
        #         "verbose": self.verbose,
        #     },
        #     os.path.join(
        #         model_path,
        #         model_name + '_metadata.json'))

    def load(self, model_path, model_name):
        """Loading the model.
        :param model_path: The path where to save it.
        :type model_path: string
        :param model_name: The name of the model.
        :type model_name: string
        """

        self.nn_model = load_keras_model(model_path, model_name, custom_loss=self.cost_function)

        # dict_params = load_json(os.path.join(
        #     model_path,
        #     model_name + '_metadata.json'))
        # self.n_epochs = dict_params["n_epochs"]
        # self.batch_size = dict_params["batch_size"]
        # self.learning_rate = dict_params["learning_rate"]
        # self.cost_function = dict_params["cost_function"]
        # self.verbose = dict_params["verbose"]

        return self


def create_dataloader(X, y, batch_size, seed):
    # Create a TensorDataset from the tensors X and y
    dataset = TensorDataset(X, y)

    # Create a RandomSampler and provide it with a seed
    sampler = RandomSampler(dataset,
                            replacement=False,
                            num_samples=None,
                            generator=torch.Generator().manual_seed(seed))

    # Use the RandomSampler in the DataLoader
    dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler,
                            num_workers=4)

    return dataloader


class NNTorch():
    """Model to wrap around PyTorch neural networks."""

    def __init__(self,
                 nn_key='MLP',
                 nn_params=None,
                 n_epochs=10,
                 batch_size=100,
                 learning_rate=0.001,
                 cost_function=torch.nn.MSELoss(),
                 metrics=['accuracy'],
                 verbose=True):
        """Instantiate the model."""
        self.val_accuracies = None
        set_seed(42)
        self.nn_params = nn_params
        self.nn_key = nn_key
        self.nn_model = self.create_model()
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.cost_function = cost_function
        self.verbose = verbose
        self.metrics = metrics

    def create_model(self):
        nn_params = self.nn_params

        if self.nn_key == 'MLP':
            return MLPTorchModel(input_shape=nn_params['input_shape'],
                                 output_shape=nn_params['output_shape'],
                                 in_nn=nn_params['in_nn'],
                                 activations=nn_params['activations'])

        elif self.nn_key == 'CNN':
            return CNNTorch(input_shape=nn_params['input_shape'],
                            output_shape=nn_params['output_shape'],
                            filters=nn_params['filters'],
                            in_nn=nn_params['in_nn'],
                            activations=nn_params['activations'])

        elif self.nn_key == 'CNN2D':
            return CNN2DTorch(
                input_shape=nn_params['input_shape'],
                output_shape=nn_params['output_shape'],
                conv_filters=nn_params['filters'],
                fc_nodes=nn_params['fc_nodes'],
                # activation=nn_params['activation']
            )

        elif self.nn_key == 'FlexCNN2D':
            return FlexibleCNN2DTorch(
                input_shape=nn_params['input_shape'],
                output_shape=nn_params['output_shape'],
                conv_filters=nn_params['filters'],
                fc_nodes=nn_params['fc_nodes'],
                activation=nn_params['activation']
            )
        
        elif self.nn_key == 'AlexNet':
            layers_to_train = ['features.conv4', 'classifier.dense2']
            alexnet = torchvision.models.alexnet(pretrained=True)
            for name, param in alexnet.named_parameters():
                if all(not name.startswith(layer) for layer in layers_to_train):
                    param.requires_grad = False

            # Get the number of input features from the existing last layer
            num_ftrs = alexnet.classifier[6].in_features

            # Replace the final layer with a new one having the desired output size
            print(self.nn_params)  # Add this line for debugging

            alexnet.classifier[6] = torch.nn.Linear(num_ftrs, self.nn_params['output_shape'])

            return alexnet

        elif self.nn_key == 'FlexibleCNN2DTorchPower':

            return FlexibleCNN2DTorchPower(conv_module_params=nn_params['conv_module_params'],
                                           classifier_params=nn_params['classifier_params'],
                                           avgpool_output_size=nn_params['avgpool_output_size']
                                           )
        elif self.nn_key == 'my':
            return MyTorchModel(input_shape=nn_params['input_shape'],
                                output_shape=nn_params['output_shape'],
                                in_nn=nn_params['in_nn'],
                                activations=nn_params['activations'])

        else:
            raise ValueError(f"Unsupported nn_key: {self.nn_key}")

    def fit(self,  X, y, validation_data=None,# dataloader=None, validation_dataloader=None,
            continue_training=True):
        """Train the neural network."""
        if self.verbose:
            print(self.nn_model)

        if not continue_training:
            self.nn_model = self.create_model()

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.nn_model.to(device)

        if torch.cuda.is_available() and torch.cuda.device_count() > 1:
            print(f"Using {torch.cuda.device_count()} GPUs!")
            self.nn_model = torch.nn.DataParallel(self.nn_model)

        print(self.nn_model.parameters())
        optimizer_choice = optim.Adam(self.nn_model.parameters(),
                                      lr=self.learning_rate,
                                      betas=(0.9, 0.999),
                                      amsgrad=False)
        # if dataloader is None:
        #     # Convert numpy array to PyTorch Tensor
        #     X, y = torch.Tensor(X), torch.Tensor(y)
        #     dataset = TensorDataset(X, y)
        #     # dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        #     dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True,
        #                             num_workers=0)  # <--- HERE
        #     # dataloader = create_dataloader(X, y, self.batch_size, seed=1)

        X, y = torch.Tensor(X), torch.Tensor(y)
        dataset = TensorDataset(X, y)
        # dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True,
                                num_workers=0)  # <--- HERE

        if validation_data is not None:
            x_val, y_val = (torch.Tensor(validation_data[0]), torch.Tensor(validation_data[1]))
            val_dataset = TensorDataset(x_val, y_val)
            val_dataloader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=True,
                                        num_workers=0)  # <--- HERE
            # Add a new list to store validation accuracies
            self.val_accuracies = []

        # # Add a new list to store validation accuracies
        # if validation_dataloader is not None:
        #     self.val_accuracies = []

        for epoch in range(self.n_epochs):
            train_loss = 0.0
            train_correct = 0
            num_train_samples = 0

            # Learning rate adjustment
            # if (epoch + 1) % (self.n_epochs // 3) == 0:
            #     for param_group in optimizer_choice.param_groups:
            #         param_group['lr'] /= 10
            #         print(param_group['lr'])

            if epoch == 50:
                for param_group in optimizer_choice.param_groups:
                    param_group['lr'] = 0.00001
                    print(param_group['lr'])

            if epoch == 60:
                for param_group in optimizer_choice.param_groups:
                    param_group['lr'] = 0.001
                    print(param_group['lr'])

            if epoch == 70:
                for param_group in optimizer_choice.param_groups:
                    param_group['lr'] = 0.0001
                    print(param_group['lr'])
            #
            # if epoch == 11:
            #     for param_group in optimizer_choice.param_groups:
            #         param_group['lr'] = 0.001
            #         print(param_group['lr'])

            # Training
            self.nn_model.train()
            for inputs, targets in dataloader:
                inputs, targets = inputs.to(device), targets.to(device)
                optimizer_choice.zero_grad()
                outputs = self.nn_model(inputs)
                loss = self.cost_function(outputs, targets)
                loss.backward()
                optimizer_choice.step()

                batch_loss, batch_acc = calc_loss_acc(outputs, targets, self.cost_function)
                train_loss += batch_loss
                train_correct += batch_acc * targets.size(0)  # un-normalize the accuracy
                num_train_samples += targets.size(0)

            train_loss /= len(dataloader)
            train_acc = train_correct / num_train_samples

            # Validation
            if validation_data is not None:
                val_loss, val_acc = self.validate_model(self.nn_model, val_dataloader,
                                                        self.cost_function, device)
                self.val_accuracies.append(val_acc)

            # Validation
            # if validation_data is not None:
            #     val_inputs, val_targets = validation_data
            #     self.nn_model.eval()  # Set the model to evaluation mode
            #     val_outputs = self.nn_model(val_inputs)
            #     val_loss, val_acc = calc_loss_acc(val_outputs, val_targets, self.cost_function)

                # self.val_accuracies.append(val_acc)

            # Validation
            # if validation_dataloader is not None:
            #     val_loss = 0.0
            #     val_correct = 0
            #     num_val_samples = 0
            #
            #     self.nn_model.eval()  # Set the model to evaluation mode
            #     for val_inputs, val_targets in validation_dataloader:
            #         val_outputs = self.nn_model(val_inputs)
            #         batch_loss, batch_acc = calc_loss_acc(val_outputs, val_targets,
            #                                               self.cost_function)
            #
            #         val_loss += batch_loss
            #         val_correct += batch_acc * val_targets.size(0)
            #         num_val_samples += val_targets.size(0)
            #
            #     val_loss /= len(validation_dataloader)
            #     val_acc = val_correct / num_val_samples
            #
            #     self.val_accuracies.append(val_acc)

            if self.verbose:
                print(
                    f'Epoch {epoch + 1}/{self.n_epochs} Train Loss: {train_loss:.4f} Train Acc: {train_acc:.4f}',
                    end='')
                if validation_data is not None:# or validation_dataloader is not None:
                    print(f' Val Loss: {val_loss:.4f} Val Acc: {val_acc:.4f}')
                else:
                    print()

        return self

    @staticmethod
    def validate_model(nn_model, val_dataloader, cost_function, device):
        nn_model.eval()  # Set the model to evaluation mode
        val_loss = 0.0
        val_correct = 0
        num_train_samples = 0
        for inputs, targets in val_dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = nn_model(inputs)
            batch_loss, batch_acc = calc_loss_acc(outputs, targets, cost_function)
            val_loss += batch_loss
            val_correct += batch_acc * targets.size(0)  # un-normalize the accuracy
            num_train_samples += targets.size(0)

        val_loss /= len(val_dataloader)
        val_acc = val_correct / num_train_samples

        return val_loss, val_acc

    def score(self, X, y):
        # Perform evaluation using the desired metric
        # Compute and return the score
        return accuracy_score(y, self.predict(X))

    def get_params(self, deep=True):
        return self.__dict__

    def set_params(self, **params):
        for parameter, value in params.items():
            setattr(self, parameter, value)
        return self

    def save(self, model_path, model_name):
        """Saving the model."""
        if len(model_name) > 100:
            model_name = model_name[:100]

        torch.save(self.nn_model.state_dict(), os.path.join(model_path, model_name + '.pt'))
        # save additional params in a json
        params = {
            "nn_params": self.nn_params,
            "n_epochs": self.n_epochs,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "cost_function": str(self.cost_function),
            "verbose": self.verbose,
            "metrics": self.metrics
        }
        with open(os.path.join(model_path, model_name + '_params.json'), 'w') as f:
            json.dump(params, f)

    def load(self, model_path, model_name):
        """Loading the model."""
        # Shorten the model name if it's too long
        if len(model_name) > 100:
            model_name = model_name[:100]
        self.nn_model.load_state_dict(torch.load(os.path.join(model_path, model_name + '.pt')))
        # load additional params from a json
        with open(os.path.join(model_path, model_name + '_params.json'), 'r') as f:
            params = json.load(f)
        self.nn_params = params["nn_params"]
        self.n_epochs = params["n_epochs"]
        self.batch_size = params["batch_size"]
        self.learning_rate = params["learning_rate"]
        if params["cost_function"] == "MSELoss":
            self.cost_function = torch.nn.MSELoss()
        # add other cost functions as needed
        self.verbose = params["verbose"]
        self.metrics = params["metrics"]

        return self

    def predict(self, X):
        """Make predictions with the trained model."""
        self.nn_model.eval()  # Set the model to evaluation mode
        X = torch.Tensor(X)
        with torch.no_grad():  # Do not calculate gradients to speed up computation
            outputs = self.nn_model(X)
            probabilities = torch.nn.functional.softmax(outputs,
                                                        dim=1)  # Apply softmax to get probabilities
            _, predicted_classes = torch.max(probabilities,
                                             1)  # Get the class with the highest probability
        return predicted_classes.numpy()  # Convert tensor to numpy array
