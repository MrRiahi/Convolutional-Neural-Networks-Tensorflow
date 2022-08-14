import unittest
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Activation, Dense, Dropout, AveragePooling2D
from tensorflow.keras.optimizers import SGD

from ..VGG11 import VGG11


class VGG11TestCase(unittest.TestCase):
    """
    Test cases for VGG11 convolutional neural network.
    """

    def test_VGG11(self):
        """
        Test the VGG11 model layers.
        """
        n_classes = 15
        vgg11 = VGG11(input_shape=(224, 224), classes=n_classes)
        model = vgg11.vgg11()
        real_value = self.get_model_summary(model=model)

        expected_value = self.get_expected_model_config(n_classes=n_classes)

        self.assertEqual(first=real_value, second=expected_value)

    @staticmethod
    def get_model_summary(model):
        """
        Get the configuration of the model.
        :param model:
        :return:
        """

        optimizer = SGD(learning_rate=0.1)

        model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

        model_summary = []

        for layer in model.layers:
            layer_description = [layer.__class__.__name__, layer.output_shape, layer.count_params()]

            if type(layer) == Conv2D:
                layer_description.append(layer.kernel_size)
                layer_description.append(layer.strides)
                layer_description.append(layer.padding)
                layer_description.append(layer.activation.__name__)

            elif type(layer) == MaxPooling2D or type(layer) == AveragePooling2D:
                layer_description.append(layer.pool_size)
                layer_description.append(layer.strides)
                layer_description.append(layer.padding)

            elif type(layer) == Dense:
                layer_description.append(layer.activation.__name__)

            elif type(layer) == Dropout:
                layer_description.append(layer.rate)

            elif type(layer) == Activation:
                layer_description.append(layer.activation.__name__)

            model_summary.append(layer_description)

        return model_summary

    @staticmethod
    def get_expected_model_config(n_classes):
        """
        Get the expected model configuration.
        :param n_classes:
        :return:
        """

        expected_model_config = [
            # Input layer
            ['InputLayer', [(None, 224, 224, 3)], 0],
            # Conv block 1
            ['Conv2D', (None, 224, 224, 64), 1792, (3, 3), (1, 1), 'same', 'linear'],
            ['BatchNormalization', (None, 224, 224, 64), 256],
            ['Activation', (None, 224, 224, 64), 0, 'relu'],
            # Max pooling block 1
            ['MaxPooling2D', (None, 112, 112, 64), 0, (2, 2), (2, 2), 'valid'],
            # Conv block 2
            ['Conv2D', (None, 112, 112, 128), 73856, (3, 3), (1, 1), 'same', 'linear'],
            ['BatchNormalization', (None, 112, 112, 128), 512],
            ['Activation', (None, 112, 112, 128), 0, 'relu'],
            # Max block 2
            ['MaxPooling2D', (None, 56, 56, 128), 0, (2, 2), (2, 2), 'valid'],
            # Conv block 3
            ['Conv2D', (None, 56, 56, 256), 295168, (3, 3), (1, 1), 'same', 'linear'],
            ['BatchNormalization', (None, 56, 56, 256), 1024],
            ['Activation', (None, 56, 56, 256), 0, 'relu'],
            ['Conv2D', (None, 56, 56, 256), 590080, (3, 3), (1, 1), 'same', 'linear'],
            ['BatchNormalization', (None, 56, 56, 256), 1024],
            ['Activation', (None, 56, 56, 256), 0, 'relu'],
            # Max block 3
            ['MaxPooling2D', (None, 28, 28, 256), 0, (2, 2), (2, 2), 'valid'],
            # Conv block 4
            ['Conv2D', (None, 28, 28, 512), 1180160, (3, 3), (1, 1), 'same', 'linear'],
            ['BatchNormalization', (None, 28, 28, 512), 2048],
            ['Activation', (None, 28, 28, 512), 0, 'relu'],
            ['Conv2D', (None, 28, 28, 512), 2359808, (3, 3), (1, 1), 'same', 'linear'],
            ['BatchNormalization', (None, 28, 28, 512), 2048],
            ['Activation', (None, 28, 28, 512), 0, 'relu'],
            # Max block 4
            ['MaxPooling2D', (None, 14, 14, 512), 0, (2, 2), (2, 2), 'valid'],
            # Conv block 5
            ['Conv2D', (None, 14, 14, 512), 2359808, (3, 3), (1, 1), 'same', 'linear'],
            ['BatchNormalization', (None, 14, 14, 512), 2048],
            ['Activation', (None, 14, 14, 512), 0, 'relu'],
            ['Conv2D', (None, 14, 14, 512), 2359808, (3, 3), (1, 1), 'same', 'linear'],
            ['BatchNormalization', (None, 14, 14, 512), 2048],
            ['Activation', (None, 14, 14, 512), 0, 'relu'],
            # Max block 5
            ['MaxPooling2D', (None, 7, 7, 512), 0, (2, 2), (2, 2), 'valid'],
            # Flatten
            ['Flatten', (None, 25088), 0],
            # Dense 1
            ['Dense', (None, 4096), 102764544, 'relu'],
            ['Dropout', (None, 4096), 0, 0.5],
            # Dense 2
            ['Dense', (None, 4096), 16781312, 'relu'],
            ['Dropout', (None, 4096), 0, 0.5],
            # Softmax
            ['Dense', (None, n_classes), (4096 + 1) * n_classes, 'softmax']
        ]
        return expected_model_config
