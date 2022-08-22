from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, Input, \
                                    Dense, Flatten, Add, MaxPooling2D, AveragePooling2D, Concatenate, Dropout
from tensorflow.keras.initializers import random_uniform
from tensorflow.keras.models import Model


class InceptionResNetV1:
    """
    This class is the implementation of the Inception-ResNetV1 convolutional neural network.
    This network consist of six parts:
        1) stem
        2) inception_resnet_v1_A
        3) inception_resnet_v1_B
        4) inception_resnet_v1_C
        5) reduction_A
        6) reduction_B
    """

    def __init__(self, input_shape, classes):
        self.input_shape = input_shape
        self.classes = classes

    @staticmethod
    def __stem(X):
        """
        The stem block of the Inception-ResNetV1.
        :param X:
        :return:
        """

        X = Conv2D(filters=32, kernel_size=(3, 3), strides=(2, 2), padding='valid',
                   kernel_initializer=random_uniform)(X)
        X = BatchNormalization()(X)
        X = Activation('relu')(X)

        X = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='valid',
                   kernel_initializer=random_uniform)(X)
        X = BatchNormalization()(X)
        X = Activation('relu')(X)

        X = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same',
                   kernel_initializer=random_uniform)(X)
        X = BatchNormalization()(X)
        X = Activation('relu')(X)

        X = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='valid')(X)

        X = Conv2D(filters=80, kernel_size=(1, 1), strides=(1, 1), padding='same',
                   kernel_initializer=random_uniform)(X)
        X = BatchNormalization()(X)
        X = Activation('relu')(X)

        X = Conv2D(filters=192, kernel_size=(3, 3), strides=(1, 1), padding='valid',
                   kernel_initializer=random_uniform)(X)
        X = BatchNormalization()(X)
        X = Activation('relu')(X)

        X = Conv2D(filters=256, kernel_size=(3, 3), strides=(2, 2), padding='valid',
                   kernel_initializer=random_uniform)(X)
        X = BatchNormalization()(X)
        X = Activation('relu')(X)

        return X

    @staticmethod
    def __inception_resnet_a(X, scale=0.1):
        """
        Build the Inception-ResNet-A block of the Inception-ResNetV1.
        :param X:
        :param scale:
        :return:
        """

        # first branch
        X_b1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same',
                      kernel_initializer=random_uniform)(X)
        X_b1 = BatchNormalization()(X_b1)
        X_b1 = Activation('relu')(X_b1)

        # second branch
        X_b2 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same',
                      kernel_initializer=random_uniform)(X)
        X_b2 = BatchNormalization()(X_b2)
        X_b2 = Activation('relu')(X_b2)

        X_b2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same',
                      kernel_initializer=random_uniform)(X_b2)
        X_b2 = BatchNormalization()(X_b2)
        X_b2 = Activation('relu')(X_b2)

        # third branch
        X_b3 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same',
                      kernel_initializer=random_uniform)(X)
        X_b3 = BatchNormalization()(X_b3)
        X_b3 = Activation('relu')(X_b3)

        X_b3 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same',
                      kernel_initializer=random_uniform)(X_b3)
        X_b3 = BatchNormalization()(X_b3)
        X_b3 = Activation('relu')(X_b3)

        X_b3 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same',
                      kernel_initializer=random_uniform)(X_b3)
        X_b3 = BatchNormalization()(X_b3)
        X_b3 = Activation('relu')(X_b3)

        # concat layers
        X_concat = Concatenate(axis=-1)([X_b1, X_b2, X_b3])

        # linear conv layer
        X_shortcut = Conv2D(filters=256, kernel_size=(1, 1), strides=(1, 1), padding='same',
                            kernel_initializer=random_uniform)(X_concat)
        X_shortcut = BatchNormalization()(X_shortcut)

        X = Add()([X, scale * X_shortcut])

        return X

    @staticmethod
    def __reduction_a(X):
        """
        Build the Reduction-A block of the Inception-ResNetV1 network.
        :param X:
        :return:
        """

        # first branch
        X_b1 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='valid')(X)

        # second branch
        X_b2 = Conv2D(filters=384, kernel_size=(3, 3), strides=(2, 2), padding='valid',
                      kernel_initializer=random_uniform)(X)
        X_b2 = BatchNormalization()(X_b2)
        X_b2 = Activation('relu')(X_b2)

        # third branch
        X_b3 = Conv2D(filters=192, kernel_size=(1, 1), strides=(1, 1), padding='same',
                      kernel_initializer=random_uniform)(X)
        X_b3 = BatchNormalization()(X_b3)
        X_b3 = Activation('relu')(X_b3)

        X_b3 = Conv2D(filters=192, kernel_size=(1, 1), strides=(1, 1), padding='same',
                      kernel_initializer=random_uniform)(X_b3)
        X_b3 = BatchNormalization()(X_b3)
        X_b3 = Activation('relu')(X_b3)

        X_b3 = Conv2D(filters=256, kernel_size=(3, 3), strides=(2, 2), padding='valid',
                      kernel_initializer=random_uniform)(X_b3)
        X_b3 = BatchNormalization()(X_b3)
        X_b3 = Activation('relu')(X_b3)

        X = Concatenate(axis=-1)([X_b1, X_b2, X_b3])

        return X

    @staticmethod
    def __inception_resnet_b(X, scale=0.1):
        """
        Build the Inception-ResNet-B block of the Inception-ResNetV1.
        :param X:
        :param scale:
        :return:
        """

        # first branch
        X_b1 = Conv2D(filters=128, kernel_size=(1, 1), strides=(1, 1), padding='same',
                      kernel_initializer=random_uniform)(X)
        X_b1 = BatchNormalization()(X_b1)
        X_b1 = Activation('relu')(X_b1)

        # second branch
        X_b2 = Conv2D(filters=128, kernel_size=(1, 1), strides=(1, 1), padding='same',
                      kernel_initializer=random_uniform)(X)
        X_b2 = BatchNormalization()(X_b2)
        X_b2 = Activation('relu')(X_b2)

        X_b2 = Conv2D(filters=128, kernel_size=(1, 7), strides=(1, 1), padding='same',
                      kernel_initializer=random_uniform)(X_b2)
        X_b2 = BatchNormalization()(X_b2)
        X_b2 = Activation('relu')(X_b2)

        X_b2 = Conv2D(filters=128, kernel_size=(7, 1), strides=(1, 1), padding='same',
                      kernel_initializer=random_uniform)(X_b2)
        X_b2 = BatchNormalization()(X_b2)
        X_b2 = Activation('relu')(X_b2)

        # concat layers
        X_concat = Concatenate(axis=-1)([X_b1, X_b2])

        # linear conv layer
        X_shortcut = Conv2D(filters=896, kernel_size=(1, 1), strides=(1, 1), padding='same',
                            kernel_initializer=random_uniform)(X_concat)
        X_shortcut = BatchNormalization()(X_shortcut)

        X = Add()([X, scale * X_shortcut])

        return X

    @staticmethod
    def __reduction_b(X):
        """
        Build the Reduction-B block of the Inception-ResNetV1 network.
        :param X:
        :return:
        """

        # first branch
        X_b1 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='valid')(X)

        # second branch
        X_b2 = Conv2D(filters=256, kernel_size=(1, 1), strides=(1, 1), padding='same',
                      kernel_initializer=random_uniform)(X)
        X_b2 = BatchNormalization()(X_b2)
        X_b2 = Activation('relu')(X_b2)

        X_b2 = Conv2D(filters=384, kernel_size=(3, 3), strides=(2, 2), padding='valid',
                      kernel_initializer=random_uniform)(X_b2)
        X_b2 = BatchNormalization()(X_b2)
        X_b2 = Activation('relu')(X_b2)

        # third branch
        X_b3 = Conv2D(filters=256, kernel_size=(1, 1), strides=(1, 1), padding='same',
                      kernel_initializer=random_uniform)(X)
        X_b3 = BatchNormalization()(X_b3)
        X_b3 = Activation('relu')(X_b3)

        X_b3 = Conv2D(filters=256, kernel_size=(3, 3), strides=(2, 2), padding='valid',
                      kernel_initializer=random_uniform)(X_b3)
        X_b3 = BatchNormalization()(X_b3)
        X_b3 = Activation('relu')(X_b3)

        # forth branch
        X_b4 = Conv2D(filters=256, kernel_size=(1, 1), strides=(1, 1), padding='same',
                      kernel_initializer=random_uniform)(X)
        X_b4 = BatchNormalization()(X_b4)
        X_b4 = Activation('relu')(X_b4)

        X_b4 = Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same',
                      kernel_initializer=random_uniform)(X_b4)
        X_b4 = BatchNormalization()(X_b4)
        X_b4 = Activation('relu')(X_b4)

        X_b4 = Conv2D(filters=256, kernel_size=(3, 3), strides=(2, 2), padding='valid',
                      kernel_initializer=random_uniform)(X_b4)
        X_b4 = BatchNormalization()(X_b4)
        X_b4 = Activation('relu')(X_b4)

        X = Concatenate(axis=-1)([X_b1, X_b2, X_b3, X_b4])

        return X

    @staticmethod
    def __inception_resnet_c(X, scale=0.1):
        """
        Build the Inception-ResNet-C block of the Inception-ResNetV1.
        :param X:
        :param scale:
        :return:
        """

        # first branch
        X_b1 = Conv2D(filters=192, kernel_size=(1, 1), strides=(1, 1), padding='same',
                      kernel_initializer=random_uniform)(X)
        X_b1 = BatchNormalization()(X_b1)
        X_b1 = Activation('relu')(X_b1)

        # second branch
        X_b2 = Conv2D(filters=192, kernel_size=(1, 1), strides=(1, 1), padding='same',
                      kernel_initializer=random_uniform)(X)
        X_b2 = BatchNormalization()(X_b2)
        X_b2 = Activation('relu')(X_b2)

        X_b2 = Conv2D(filters=192, kernel_size=(1, 3), strides=(1, 1), padding='same',
                      kernel_initializer=random_uniform)(X_b2)
        X_b2 = BatchNormalization()(X_b2)
        X_b2 = Activation('relu')(X_b2)

        X_b2 = Conv2D(filters=192, kernel_size=(3, 1), strides=(1, 1), padding='same',
                      kernel_initializer=random_uniform)(X_b2)
        X_b2 = BatchNormalization()(X_b2)
        X_b2 = Activation('relu')(X_b2)

        # concat layers
        X_concat = Concatenate(axis=-1)([X_b1, X_b2])

        # linear conv layer
        X_shortcut = Conv2D(filters=1792, kernel_size=(1, 1), strides=(1, 1), padding='same',
                            kernel_initializer=random_uniform)(X_concat)
        X_shortcut = BatchNormalization()(X_shortcut)

        X = Add()([X, scale * X_shortcut])

        return X

    def __call__(self):
        """
        Build the Inception-ResNetV1 network.
        """

        X_input = Input(shape=(self.input_shape[0], self.input_shape[1], 3))

        # stem layer
        X = self.__stem(X=X_input)

        # Inception-ResNet-A blocks
        for i in range(5):
            X = self.__inception_resnet_a(X=X, scale=0.1)

        # Reduction-A block
        X = self.__reduction_a(X=X)

        # Inception-ResNet-B blocks
        for i in range(10):
            X = self.__inception_resnet_b(X=X, scale=0.1)

        # Reduction-B block
        X = self.__reduction_b(X=X)

        # Inception-ResNet-C blocks
        for i in range(5):
            X = self.__inception_resnet_c(X=X, scale=0.1)

        # Average pooling
        X = AveragePooling2D(pool_size=(8, 8), strides=(1, 1), padding='valid')(X)

        # Classifier part
        X = Flatten()(X)
        X = Dropout(rate=0.2)(X)
        X_output = Dense(units=self.classes, activation='softmax', name='output',
                         kernel_initializer=random_uniform)(X)

        # Create model
        model = Model(inputs=X_input, outputs=X_output)

        return model
