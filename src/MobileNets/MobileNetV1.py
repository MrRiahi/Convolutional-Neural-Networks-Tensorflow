from tensorflow.keras.layers import Conv2D, BatchNormalization, DepthwiseConv2D, Activation, Input, \
                                    AveragePooling2D, Dense, Flatten
from tensorflow.keras.initializers import random_uniform
from tensorflow.keras.models import Model


class MobileNetV1:

    def __init__(self, input_shape, classes):
        self.input_shape = input_shape
        self.classes = classes

    @staticmethod
    def __mobile_net_block(X, filters, stride, f=3, initializer=random_uniform):
        """
        Implementation of the mobile_net block in MobileNetV1.
        :param X: input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)
        :param f: shape of the conv window
        :param filters: number of filters in mobile_net block
        :param stride: shape of stride in conv layers
        :param initializer: to set up the initial weights of a layer. Equals to random uniform initializer
        :return: X output of the mobile_net block which is a tensor of shape (m, n_H, n_W, n_C)
        """

        # Depth wise part
        X = DepthwiseConv2D(kernel_size=(f, f), strides=(stride, stride), padding='same',
                            depthwise_initializer=initializer)(X)
        X = BatchNormalization()(X)
        X = Activation('relu')(X)

        # Separable part
        X = Conv2D(filters=filters, kernel_size=(1, 1), strides=(1, 1), padding='same',
                   kernel_initializer=initializer)(X)
        X = BatchNormalization()(X)
        X = Activation('relu')(X)

        return X

    def mobile_net_v1(self):
        """
        Build MobileNet version 1 network
        :return:
        """

        # Define the input as a tensor with shape input_shape
        X_input = Input(shape=(self.input_shape[0], self.input_shape[1], 3))

        # Layer 1
        X = Conv2D(kernel_size=(3, 3), filters=32, strides=(2, 2), padding='same',
                   kernel_initializer=random_uniform)(X_input)
        X = BatchNormalization()(X)

        # Layer 2
        X = self.__mobile_net_block(X=X, filters=64, stride=1, f=3, initializer=random_uniform)

        # Layer 3
        X = self.__mobile_net_block(X=X, filters=128, stride=2, f=3, initializer=random_uniform)

        # Layer 4
        X = self.__mobile_net_block(X=X, filters=128, stride=1, f=3, initializer=random_uniform)

        # Layer 5
        X = self.__mobile_net_block(X=X, filters=256, stride=2, f=3, initializer=random_uniform)

        # Layer 6
        X = self.__mobile_net_block(X=X, filters=256, stride=1, f=3, initializer=random_uniform)

        # Layer 7
        X = self.__mobile_net_block(X=X, filters=256, stride=2, f=3, initializer=random_uniform)

        # Layer 8
        X = self.__mobile_net_block(X=X, filters=512, stride=1, f=3, initializer=random_uniform)

        # Layer 9
        X = self.__mobile_net_block(X=X, filters=512, stride=1, f=3, initializer=random_uniform)

        # Layer 10
        X = self.__mobile_net_block(X=X, filters=512, stride=1, f=3, initializer=random_uniform)

        # Layer 11
        X = self.__mobile_net_block(X=X, filters=512, stride=1, f=3, initializer=random_uniform)

        # Layer 12
        X = self.__mobile_net_block(X=X, filters=512, stride=1, f=3, initializer=random_uniform)

        # Layer 13
        X = self.__mobile_net_block(X=X, filters=512, stride=2, f=3, initializer=random_uniform)

        # Layer 14
        X = self.__mobile_net_block(X=X, filters=1024, stride=1, f=3, initializer=random_uniform)

        # Average pooling layer
        X = AveragePooling2D(pool_size=(7, 7), strides=(1, 1))(X)

        # Output layer
        X = Flatten()(X)
        X = Dense(self.classes, activation='softmax')(X)

        # Create model
        model = Model(inputs=X_input, outputs=X)

        return model
