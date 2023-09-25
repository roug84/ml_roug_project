"""
This file contains the MLP TensorFlow neural network model class
"""

# import tensorflow as tf
import torch
import numpy as np


# class ActivationModule(torch.nn.Module):
#     def __init__(self, func):
#         super(ActivationModule, self).__init__()
#         self.func = func
#
#     def forward(self, x):
#         return self.func(x)
#
# # Modify the ACTIVATION_FUNCTIONS dictionary to use these wrappers:
# ACTIVATION_FUNCTIONS = {
#     'relu': ActivationModule(torch.relu),
#     'tanh': ActivationModule(torch.tanh),
#     'identity': ActivationModule(lambda x: x)  # No-op for identity
# }
class LinearActivation(torch.nn.Module):
    def forward(self, x):
        return x


ACTIVATION_FUNCTIONS = {
    'relu': torch.nn.ReLU(),
    'tanh': torch.nn.Tanh(),
    'sigmoid': torch.nn.Sigmoid(),
    'identity': torch.nn.Identity(),
    'linear': LinearActivation()
}
#
# class MLPModel(tf.keras.Sequential):
#     def __init__(self, input_shape, output_shape, in_nn=[40, 150], activations=['tanh', 'relu']):
#         """
#         This class is a MLP TensorFlow model
#         :param input_shape: shape of the input layer
#         :type input_shape: int
#         :param output_shape: shape of the output layer
#         :type output_shape: int
#         :param in_nn: number of neurons in each layer
#         :type in_nn: list
#         :param activations: activation function in each layer
#         :type activations: list
#         """
#         super().__init__()
#         self.__class__.__name__ = "Sequential"
#
#         for i in range(0, len(in_nn)):
#             # add layer
#             params_tmp = {'units': in_nn[i], 'activation': activations[i],
#                           'kernel_initializer': "uniform"}
#             if i == 0:
#                 params_tmp['input_shape'] = (None, input_shape)
#             self.add(tf.keras.layers.Dense(**params_tmp))
#         # add output layer
#         self.add(tf.keras.layers.Dense(output_shape, activation='softmax'))
#
#
# class AEModel(tf.keras.Sequential):
#     def __init__(self, input_shape, output_shape, in_nn=[40, 150], activations=['tanh', 'relu']):
#         """
#         This class is a MLP TensorFlow model
#         :param input_shape: shape of the input layer
#         :type input_shape: int
#         :param output_shape: shape of the output layer
#         :type output_shape: int
#         :param in_nn: number of neurons in each layer
#         :type in_nn: list
#         :param activations: activation function in each layer
#         :type activations: list
#         """
#         super().__init__()
#         self.__class__.__name__ = "Sequential"
#
#         for i in range(0, len(in_nn)):
#             # add layer
#             params_tmp = {'units': in_nn[i], 'activation': activations[i],
#                           'kernel_initializer': "uniform"}
#             if i == 0:
#                 params_tmp['input_shape'] = (None, input_shape)
#             self.add(tf.keras.layers.Dense(**params_tmp))
#         # add output layer
#         self.add(tf.keras.layers.Dense(input_shape, activation='tanh'))
#


class MLPTorchModel(torch.nn.Module):
    def __init__(self, input_shape, output_shape, in_nn=[40, 150], activations=['tanh', 'relu']):
        super().__init__()

        layers = []
        previous_shape = input_shape
        for i in range(len(in_nn)):
            layers.append(torch.nn.Linear(previous_shape, in_nn[i]))

            if activations[i] == 'relu':
                layers.append(torch.nn.ReLU())
            elif activations[i] == 'tanh':
                layers.append(torch.nn.Tanh())

            previous_shape = in_nn[i]

        layers.append(torch.nn.Linear(previous_shape, output_shape))
        self.layers = torch.nn.Sequential(*layers)

    @staticmethod
    def weights_init(m):
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)

    def get_weights(self):
        """
        :returns: the weights of the network
        :rtype: list of torch float tensors
        """
        weights = []
        for params in self.parameters():
            weights.append(params.data)
        return weights

    def set_weights(self, weights):
        """
        Sets the weights of the model
        :param weights: the new weights
        :type weights: list of float tensors, as returned by get_weights
        """
        for i, param in enumerate(self.parameters()):
            param.data = weights[i].float()

    def forward(self, x):
        x = self.layers(x)
        return x


class CNNTorch(torch.nn.Module):
    def __init__(self, input_shape, output_shape, filters, in_nn, activations):
        super(CNNTorch, self).__init__()

        assert len(activations) == len(
            in_nn), "Mismatch in number of activations and in_nn layers"

        self.input_shape = input_shape
        self.output_shape = output_shape

        # Convolutional Layer
        self.conv1 = torch.nn.Conv1d(1, filters, kernel_size=1)
        flattened_size = self.conv1(torch.zeros(1, 1, self.input_shape)).view(1, -1).shape[1]

        # Dynamic creation of linear layers followed by activation layers
        layers = []
        in_features = flattened_size
        for i, (out_features, activation) in enumerate(zip(in_nn, activations)):
            layers.append(torch.nn.Linear(in_features, out_features))
            if activation not in ['identity', 'linear']:
                layers.append(ACTIVATION_FUNCTIONS[activation])
            # layers.append(torch.nn.Dropout(p=0.1))  # adding dropout here

            in_features = out_features

        self.layers = torch.nn.Sequential(*layers)

        self.d2 = torch.nn.Dropout(p=0.6)
        # Output Layer
        self.out = torch.nn.Linear(in_nn[-1], self.output_shape)

    def forward(self, x):
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float32)

        if len(x.shape) == 2:
            x = x.unsqueeze(1)

        x = torch.relu(self.conv1(x))
        x = x.view(x.size(0), -1)

        x = self.layers(x)

        x = self.d2(x)

        x = self.out(x)

        return x

class MyTorchModel(torch.nn.Module):
    def __init__(self, input_shape, output_shape, in_nn=[40, 150], activations=['tanh', 'relu']):
        super(MyTorchModel, self).__init__()
        self.f5 = torch.nn.Linear(884, 256)#(1024, 1024)

        self.conv1 = torch.nn.Conv2d(in_channels=32, out_channels=128, kernel_size=3, padding=1, stride=2)

        self.max1 = torch.nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        # self.l1 = torch.nn.GRU(input_size=64, hidden_size=64, num_layers=1, bidirectional=True,
        #                        batch_first=True)

        self.conv3 = torch.nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, padding=1,
                               stride=2)
        self.max2 = torch.nn.MaxPool2d(kernel_size=2, stride=2)

        self.flatten = torch.nn.Flatten()
        self.f1 = torch.nn.Linear(64, 128)
        self.f2 = torch.nn.Linear(128, 64)
        self.d2 = torch.nn.Dropout(p=0.6)
        self.f3 = torch.nn.Linear(64, 32)
        self.d3 = torch.nn.Dropout(p=0.7)
        self.f4 = torch.nn.Linear(32, 10)

    def forward(self, x):
        print('shape of x', x.size())
        x = self.f5(x)
        print('shape after self.f5(x)', x.size())

        x = x.view(-1, 32, 32, 1)
        print('shape after x.view(-1, 32, 32, 1)', x.size())

        x = self.conv1(x)
        print('shape after self.conv1(x)', x.size())

        # x = x.view(-1, 128, 8, 8)  # Adjust the dimensions to match max-pooling
        # print('shape after x.view(-1, 128, 8, 8)', x.size())

        x = self.max1(x)
        print('shape after self.max1(x)', x.size())

        # # After self.max1(x), add a convolutional layer to further reduce spatial dimensions
        # x = self.conv4(x)
        # print('shape after self.conv4(x)', x.size())
        # x = x.view(-1, 64, 64)  # Adjust the dimensions to match the expected input size
        # print('shape after x.view(-1, 64, input_size)', x.size())
        #
        # # x = x.view(-1, 64, 128)  # Adjust the dimensions to match the expected sequence length of 64
        # # print('shape after x.view(-1, 128, 32)', x.size())
        #
        # x, _ = self.l1(x)
        # print('shape after self.l1(x)', x.size())

        # x = x.view(-1, 128, 128, 1)
        # print('shape after x.view(-1, 128, 128, 1)', x.size())

        x = self.conv3(x)
        print('shape after self.conv3(x)', x.size())

        x = self.max2(x)
        x = self.flatten(x)
        print('shape after flattening', x.size())

        # Continue with the fully connected layers as before
        x = self.f1(x)
        x = self.f2(x)
        x = self.d2(x)
        x = self.f3(x)
        x = self.d3(x)
        y = self.f4(x)
        return y

    # class CNNTorch(torch.nn.Module):
#     def __init__(self, input_shape, output_shape, filters, in_nn, activations):
#         super(CNNTorch, self).__init__()
#
#         assert len(activations) == len(
#             in_nn), "Mismatch in number of activations and in_nn layers"
#
#         self.input_shape = input_shape
#         self.output_shape = output_shape
#
#         # Convolutional Layer
#         self.conv1 = torch.nn.Conv1d(1, filters, kernel_size=1)
#         flattened_size = self.conv1(torch.zeros(1, 1, self.input_shape)).view(1, -1).shape[1]
#         # Dynamic creation of linear layers followed by activation layers
#         layers = []
#         in_features = flattened_size
#         for i, (out_features, activation) in enumerate(zip(in_nn, activations)):
#             layers.append(torch.nn.Linear(in_features, out_features))
#             layers.append(ACTIVATION_FUNCTIONS[activation])
#             in_features = out_features
#
#         self.layers = torch.nn.Sequential(*layers)
#
#         # Output Layer
#         self.out = torch.nn.Linear(in_nn[-1], self.output_shape)
#
#     def forward(self, x):
#         if isinstance(x, np.ndarray):
#             x = torch.tensor(x, dtype=torch.float32)
#
#         if len(x.shape) == 2:
#             x = x.unsqueeze(1)
#
#         x = torch.relu(self.conv1(x))
#         x = x.view(x.size(0), -1)
#
#         x = self.layers(x)
#         x = self.out(x)
#
#         return x

# class CNNTorch(torch.nn.Module):
#     def __init__(self, input_shape, output_shape, filters=5, in_nn=None, activations=None):
#         super(CNNTorch, self).__init__()
#
#         if in_nn is None:
#             in_nn = [10, 100]
#
#         if activations is None:
#             activations = ['relu', 'tanh']
#
#         assert len(activations) == len(in_nn), "Mismatch in number of activations and in_nn layers"
#
#         self.input_shape = input_shape
#         self.output_shape = output_shape
#         self.conv1 = torch.nn.Conv1d(1, filters, kernel_size=1)
#
#         # Dummy pass to find the output shape after convolution
#         flattened_size = self.conv1(torch.zeros(1, 1, self.input_shape)).view(1, -1).shape[1]
#
#         self.layers = torch.nn.Sequential()
#         self.layers.add_module('linear0', torch.nn.Linear(flattened_size, in_nn[0]))
#         self.layers.add_module('activation0', ACTIVATION_FUNCTIONS[activations[0]])
#
#         for i in range(1, len(in_nn)):
#             self.layers.add_module('linear'+str(i), torch.nn.Linear(in_nn[i-1], in_nn[i]))
#             self.layers.add_module('activation'+str(i), ACTIVATION_FUNCTIONS[activations[i]])
#
#         self.out = torch.nn.Linear(in_nn[-1], self.output_shape)
#
#     def forward(self, x):
#         # Convert numpy array to PyTorch tensor if required
#         if isinstance(x, np.ndarray):
#             x = torch.tensor(x, dtype=torch.float32)
#
#         # Assert to check input shape. This helps in identifying shape issues.
#         assert x.shape[-1] == self.input_shape, f"Expected the last dimension to be {self.input_shape}, but got {x.shape[-1]}"
#
#         # Check if the channel dimension is already present:
#         if len(x.shape) == 2:  # No channel dimension
#             x = x.unsqueeze(1)
#
#         x = torch.relu(self.conv1(x))
#         x = x.view(x.size(0), -1)
#
#         x = self.layers(x)
#         x = self.out(x)
#
#         # Assert to check output shape. This helps in identifying issues in output.
#         assert x.shape[-1] == self.output_shape, f"Expected the last dimension of output to be {self.output_shape}, but got {x.shape[-1]}"
#
#         return x

    # def forward(self, x):
    #     x = x.view(-1, 1, self.input_shape)
    #     x = torch.relu(self.conv1(x))
    #     x = x.view(x.size(0), -1)
    #     x = self.layers(x)
    #     x = self.out(x)
    #     return x


    @staticmethod
    def weights_init(m):
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)
        elif isinstance(m, torch.nn.Conv1d):
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)


class CNN2DTorch(torch.nn.Module):
    def __init__(self, input_shape, output_shape, conv_filters=(32, 64), fc_nodes=1000):
        super(CNN2DTorch, self).__init__()

        # Convolutional layers
        self.conv1 = torch.nn.Conv2d(input_shape[0], conv_filters[0], kernel_size=5, stride=1,
                                     padding=2)
        self.pool1 = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = torch.nn.Conv2d(conv_filters[0], conv_filters[1], kernel_size=5, stride=1,
                                     padding=2)
        self.pool2 = torch.nn.MaxPool2d(kernel_size=2, stride=2)

        # Calculate flattened size for the first fully connected layer
        self.flattened_size = conv_filters[1] * (input_shape[1] // 4) * (input_shape[2] // 4)

        # Fully connected layers
        self.fc1 = torch.nn.Linear(self.flattened_size, fc_nodes)
        self.fc2 = torch.nn.Linear(fc_nodes, output_shape)

        # Dropout layer
        self.drop_out = torch.nn.Dropout()

    def forward(self, x):
        x = self.pool1(torch.relu(self.conv1(x)))
        x = self.pool2(torch.relu(self.conv2(x)))

        x = x.view(-1, self.flattened_size)

        x = torch.relu(self.fc1(x))
        x = self.drop_out(x)
        x = self.fc2(x)

        return x


class FlexibleCNN2DTorch(torch.nn.Module):
    def __init__(self, input_shape, output_shape, conv_filters=(32, 64), fc_nodes=(1024, 512),
                 activation='relu'):
        super(FlexibleCNN2DTorch, self).__init__()

        # Define the activation function based on the input argument
        if activation == 'relu':
            self.activation = torch.nn.ReLU()
        elif activation == 'sigmoid':
            self.activation = torch.nn.Sigmoid()
        # Add more if needed

        # Convolutional layers
        self.convs = torch.nn.ModuleList()
        in_channels = input_shape[0]
        for out_channels in conv_filters:
            self.convs.append(torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
            in_channels = out_channels
        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2)

        # Calculate flattened size for the first fully connected layer
        self.flattened_size = conv_filters[-1] * (input_shape[1] // (
                2**len(conv_filters))) * (input_shape[2] // (2**len(conv_filters)))

        # Fully connected layers
        self.fcs = torch.nn.ModuleList()
        in_nodes = self.flattened_size
        for out_nodes in fc_nodes:
            self.fcs.append(torch.nn.Linear(in_nodes, out_nodes))
            in_nodes = out_nodes
        self.fcs.append(torch.nn.Linear(in_nodes, output_shape))

        # Dropout layer
        self.drop_out = torch.nn.Dropout()

    def forward(self, x):
        for conv in self.convs:
            x = self.pool(self.activation(conv(x)))

        x = x.view(-1, self.flattened_size)

        for i, fc in enumerate(self.fcs):
            if i < len(self.fcs) - 1:  # Apply activation and dropout to all but the last layer
                x = self.activation(fc(x))
                x = self.drop_out(x)
            else:  # Last layer, no activation or dropout
                x = fc(x)

        return x


class FlexibleConvModule(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, pool_kernel_size,
                 pool_stride):
        super(FlexibleConvModule, self).__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                      padding=padding),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=pool_kernel_size, stride=pool_stride, padding=0)
        )

    def forward(self, x):
        return self.layers(x)


class FlexibleClassifier(torch.nn.Module):
    def __init__(self, in_features, out_features):
        super(FlexibleClassifier, self).__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Dropout(p=0.5),
            torch.nn.Linear(in_features, out_features),
            torch.nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.layers(x)


class FlexibleCNN2DTorchPower(torch.nn.Module):
    def __init__(self, conv_module_params, classifier_params, avgpool_output_size):
        super(FlexibleCNN2DTorchPower, self).__init__()

        self.conv_modules = torch.nn.ModuleList([
            FlexibleConvModule(**params) for params in conv_module_params
        ])

        self.avgpool = torch.nn.AdaptiveAvgPool2d(output_size=avgpool_output_size)

        self.classifiers = torch.nn.ModuleList([
            FlexibleClassifier(**params) for params in classifier_params[:-1]
        ])

        # Last classifier without ReLU
        self.final_classifier = torch.nn.Linear(classifier_params[-1]['in_features'],
                                          classifier_params[-1]['out_features'])

    def forward(self, x):
        for module in self.conv_modules:
            x = module(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)  # Flatten tensor

        for classifier in self.classifiers:
            x = classifier(x)

        x = self.final_classifier(x)

        return x
# class MLPTensorFlowDropOut(tf.keras.Sequential):
#     def __init__(self, input_shape, output_shape, in_nn=[10, 100],  activations=[None, 'tanh'], dropout=0.1,
#                  nn_last_dense=10):
#         """
#         This class is a MLP TensorFlow model with Dropout
#         :param input_shape: shape of the input layer
#         :type input_shape: int
#         :param output_shape: shape of the output layer
#         :type output_shape: int
#         :param in_nn: number of neurons in each layer
#         :type in_nn: list
#         :param activations: activation function in each layer
#         :type activations: list
#         :param dropout: dropout rate in the hidden layers
#         :type dropout: float
#         :param nn_last_dense: number of neurons in the last dense layer
#         :type nn_last_dense: int
#         """
#         super().__init__()
#         self.__class__.__name__ = "Sequential"
#
#         for i in range(0, len(in_nn)):
#             # add layer
#             params_tmp = {'units': in_nn[i], 'activation': activations[i], 'kernel_initializer': "uniform"}
#             if i == 0:
#                 params_tmp['input_shape'] = input_shape
#             params_tmp['name'] = 'Dense_' + str(i) + '_' + str(params_tmp['units'])
#             self.add(tf.keras.layers.Dense(**params_tmp))
#             if i > 0:
#                 self.add(tf.keras.layers.Dropout(dropout, name='Dropout_' + str(i) + str(0.1)))
#         # last dense without Dropout
#         params_tmp = {'units': nn_last_dense, 'name': 'Last_Dense'}
#         self.add(tf.keras.layers.Dense(**params_tmp))
#         # add output layer
#         self.add(tf.keras.layers.Dense(output_shape))


if __name__ == '__main__':
    nn_model = MLPModel(input_shape=10, output_shape=5)
    print(nn_model.summary())


