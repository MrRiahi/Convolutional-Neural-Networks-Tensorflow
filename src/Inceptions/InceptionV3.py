from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Activation, \
    Flatten, Dense, Input, AveragePooling2D, concatenate, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.initializers import random_uniform


class InceptionV3:
    """
    Implementation of the second version of the Inception network.
    """

    def __init__(self, input_shape, classes):
        self.input_shape = input_shape
        self.classes = classes

    @staticmethod
    def __factorized_module():
        """
        Build the factorized module. This module has two 3x3 filters instead of one 5x5 filter.
        """
        pass

    def inception_v3(self):
        """
        Build the inception v2 model.
        """

        X_input = Input(shape=(self.input_shape[0], self.input_shape[1], 3))

        # layer 1
        X = Conv2D(filters=32, kernel_size=(3, 3), strides=(2, 2), padding='valid',
                   kernel_initializer=random_uniform)(X_input)
        X = BatchNormalization()(X)
        X = Activation('relu')(X)

        # layer 2
        X = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='valid',
                   kernel_initializer=random_uniform)(X)
        X = BatchNormalization()(X)
        X = Activation('relu')(X)

        # layer 3
        X = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same',
                   kernel_initializer=random_uniform)(X)
        X = BatchNormalization()(X)
        X = Activation('relu')(X)

        # layer 4
        X = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='valid')(X)

        # layer 5
        X = Conv2D(filters=80, kernel_size=(3, 3), strides=(1, 1), padding='valid',
                   kernel_initializer=random_uniform)(X)
        X = BatchNormalization()(X)
        X = Activation('relu')(X)

        # layer 6
        X = Conv2D(filters=192, kernel_size=(3, 3), strides=(2, 2), padding='valid',
                   kernel_initializer=random_uniform)(X)
        X = BatchNormalization()(X)
        X = Activation('relu')(X)

        return X
