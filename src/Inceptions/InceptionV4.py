from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Activation, \
    Flatten, Dense, Input, AveragePooling2D, concatenate, BatchNormalization, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.initializers import random_uniform


class InceptionV4:
    """
    Implementation of the forth version of the inception network.
    This network consist of six parts:
        1) stem
        2) inception_A
        3) inception_B
        4) inception_C
        5) reduction_A
        6) reduction_B
    """

    def __init__(self, input_shape, classes):
        self.input_shape = input_shape
        self.classes = classes

    @staticmethod
    def __stem(X):
        """
        Build the stem block of the InceptionV4 network.
        :param X:
        :return:
        """

        # main branch
        X = Conv2D(filters=32, kernel_size=(3, 3), strides=(2, 2), padding='valid',
                   kernel_initializer=random_uniform)(X)

        X = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='valid',
                   kernel_initializer=random_uniform)(X)

        X = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same',
                   kernel_initializer=random_uniform)(X)

        # first pooling and conv2d parallel branches
        X_pooling = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='valid')(X)

        X_conv2d = Conv2D(filters=96, kernel_size=(3, 3), strides=(2, 2), padding='valid',
                          kernel_initializer=random_uniform)(X)

        X = Concatenate(axis=-1)([X_pooling, X_conv2d])

        # two parallel conv branches
        X_b1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same',
                      kernel_initializer=random_uniform)(X)
        X_b1 = Conv2D(filters=96, kernel_size=(3, 3), strides=(1, 1), padding='valid',
                      kernel_initializer=random_uniform)(X_b1)

        X_b2 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same',
                      kernel_initializer=random_uniform)(X)
        X_b2 = Conv2D(filters=64, kernel_size=(7, 1), strides=(1, 1), padding='same',
                      kernel_initializer=random_uniform)(X_b2)
        X_b2 = Conv2D(filters=64, kernel_size=(1, 7), strides=(1, 1), padding='same',
                      kernel_initializer=random_uniform)(X_b2)
        X_b2 = Conv2D(filters=96, kernel_size=(3, 3), strides=(1, 1), padding='valid',
                      kernel_initializer=random_uniform)(X_b2)

        X = Concatenate(axis=-1)([X_b1, X_b2])

        # second pooling and conv2d parallel branches
        X_pooling = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='valid')(X)

        X_conv2d = Conv2D(filters=192, kernel_size=(3, 3), strides=(2, 2), padding='valid',
                          kernel_initializer=random_uniform)(X)

        X = Concatenate(axis=-1)([X_pooling, X_conv2d])

        return X

    @staticmethod
    def __inception_a(X):
        """
        Build the inception-A block of the InceptionV4 network.
        :param X:
        :return:
        """

        # first branch
        X_b1 = AveragePooling2D(pool_size=(3,3), strides=(1,1), padding='same')(X)
        X_b1 = Conv2D(filters=96, kernel_size=(1, 1), strides=(1, 1), padding='same',
                      kernel_initializer=random_uniform)(X_b1)

        # second branch
        X_b2 = Conv2D(filters=96, kernel_size=(1, 1), strides=(1, 1), padding='same',
                      kernel_initializer=random_uniform)(X)

        # third branch
        X_b3 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same',
                      kernel_initializer=random_uniform)(X)
        X_b3 = Conv2D(filters=96, kernel_size=(3, 3), strides=(1, 1), padding='same',
                      kernel_initializer=random_uniform)(X_b3)

        # forth branch
        X_b4 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same',
                      kernel_initializer=random_uniform)(X)
        X_b4 = Conv2D(filters=96, kernel_size=(3, 3), strides=(1, 1), padding='same',
                      kernel_initializer=random_uniform)(X_b4)
        X_b4 = Conv2D(filters=96, kernel_size=(3, 3), strides=(1, 1), padding='same',
                      kernel_initializer=random_uniform)(X_b4)

        X = Concatenate(axis=-1)([X_b1, X_b2, X_b3, X_b4])

        return X

    @staticmethod
    def __reduction_a(X):
        """
        Build the Reduction-A block of the InceptionV4 network.
        :param X:
        :return:
        """

        # first branch
        X_b1 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='valid')(X)

        # second branch
        X_b2 = Conv2D(filters=384, kernel_size=(3, 3), strides=(2, 2), padding='valid',
                      kernel_initializer=random_uniform)(X)

        # third branch
        X_b3 = Conv2D(filters=192, kernel_size=(1, 1), strides=(1, 1), padding='same',
                      kernel_initializer=random_uniform)(X)
        X_b3 = Conv2D(filters=224, kernel_size=(1, 1), strides=(1, 1), padding='same',
                      kernel_initializer=random_uniform)(X_b3)
        X_b3 = Conv2D(filters=256, kernel_size=(3, 3), strides=(2, 2), padding='valid',
                      kernel_initializer=random_uniform)(X_b3)

        X = Concatenate(axis=-1)([X_b1, X_b2, X_b3])

        return X

    @staticmethod
    def __inception_b(X):
        """
        Build the inception-B block of the InceptionV4 network.
        :param X:
        :return:
        """

        # first branch
        X_b1 = AveragePooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(X)
        X_b1 = Conv2D(filters=128, kernel_size=(1, 1), strides=(1, 1), padding='same',
                      kernel_initializer=random_uniform)(X_b1)

        # second branch
        X_b2 = Conv2D(filters=384, kernel_size=(1, 1), strides=(1, 1), padding='same',
                      kernel_initializer=random_uniform)(X)

        # third branch
        X_b3 = Conv2D(filters=192, kernel_size=(1, 1), strides=(1, 1), padding='same',
                      kernel_initializer=random_uniform)(X)
        X_b3 = Conv2D(filters=224, kernel_size=(1, 7), strides=(1, 1), padding='same',
                      kernel_initializer=random_uniform)(X_b3)
        X_b3 = Conv2D(filters=256, kernel_size=(1, 7), strides=(1, 1), padding='same',
                      kernel_initializer=random_uniform)(X_b3)

        # forth branch
        X_b3 = Conv2D(filters=192, kernel_size=(1, 1), strides=(1, 1), padding='same',
                      kernel_initializer=random_uniform)(X)
        X_b3 = Conv2D(filters=192, kernel_size=(1, 7), strides=(1, 1), padding='same',
                      kernel_initializer=random_uniform)(X_b3)
        X_b3 = Conv2D(filters=224, kernel_size=(7, 1), strides=(1, 1), padding='same',
                      kernel_initializer=random_uniform)(X_b3)
        X_b3 = Conv2D(filters=224, kernel_size=(1, 7), strides=(1, 1), padding='same',
                      kernel_initializer=random_uniform)(X_b3)
        X_b3 = Conv2D(filters=256, kernel_size=(7, 1), strides=(1, 1), padding='same',
                      kernel_initializer=random_uniform)(X_b3)

        X = Concatenate(axis=-1)([X_b1, X_b2, X_b3])

        return X

    @staticmethod
    def __reduction_b(X):
        """
        Build the Reduction-B block of the InceptionV4 network.
        :param X:
        :return:
        """

    def inception_v4(self):
        """
        Build the InceptionV4 network.
        """

        X_input = Input(shape=(self.input_shape[0], self.input_shape[1], 3))

        # stem layer
        X = self.__stem(X=X_input)

        # Inception-A blocks
        X = self.__inception_a(X=X)
        X = self.__inception_a(X=X)
        X = self.__inception_a(X=X)
        X = self.__inception_a(X=X)

        # Reduction-A block
        X = self.__reduction_a(X=X)

        # Inception-B blocks
        X = self.__inception_b(X=X)
        X = self.__inception_b(X=X)
        X = self.__inception_b(X=X)
        X = self.__inception_b(X=X)
        X = self.__inception_b(X=X)
        X = self.__inception_b(X=X)
        X = self.__inception_b(X=X)

        # Reduction-B


        return X












