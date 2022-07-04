from tensorflow.keras.layers import Conv2D, BatchNormalization, DepthwiseConv2D, Activation, Input, \
                                    AveragePooling2D, Dense, Flatten, Add
from tensorflow.keras.initializers import random_uniform
from tensorflow.keras.models import Model


class MobileNetV2:
    """
    This class is the implementation of the second version of MobileNet which is called MobileNetV2.
    """

    def __init__(self, input_shape, classes):
        """
        Initialization of MobileNetV2 class
        :param input_shape:
        :param classes:
        """

        self.input_shape = input_shape
        self.class_numbers = classes

    @staticmethod
    def __get_number_of_expansion_filters(X, expansion_ratio):
        """
        This method calculates the number of filters for expansion part.
        :param X:
        :param expansion_ratio:
        :return:
        """

        return X.shape[-1] * expansion_ratio

    def __bottleneck_block(self, X_input, filters, expansion_ratio, filter_size=(3, 3), strides=(1, 1),
                           is_add=False, initializer=random_uniform):
        """
        This is the bottleneck block in MobileNetV2.
        The other name of this block is "Inverted Residual Block"
        :param X_input:
        :param filters:
        :param expansion_ratio:
        :param filter_size:
        :param strides:
        :param is_add:
        :param initializer:
        :return:
        """

        # Get number of expansion filters
        expansion_filters = self.__get_number_of_expansion_filters(X=X_input, expansion_ratio=expansion_ratio)

        # Expansion part
        X = Conv2D(filters=expansion_filters, kernel_size=(1, 1), strides=(1, 1), padding='same',
                   kernel_initializer=initializer)(X_input)
        X = BatchNormalization()(X)
        X = Activation('relu')(X)

        # Depth wise part
        X = DepthwiseConv2D(kernel_size=filter_size, strides=strides,
                            padding='same', depthwise_initializer=initializer)(X)
        X = BatchNormalization()(X)
        X = Activation('relu')(X)  # Update the relu activation function to relu6

        # Separable part
        X = Conv2D(filters=filters, kernel_size=(1, 1), strides=(1, 1), padding='same',
                   kernel_initializer=initializer)(X)
        X = BatchNormalization()(X)
        X = Activation('relu')(X)

        # Add layers
        if is_add:
            X = Add()([X, X_input])

        return X

    def mobile_net_v2(self):
        """
        Implementation of the MobileNetV2 network.
        :return:
        """

        # Define the input as a tensor with shape input_shape
        X_input = Input(shape=(self.input_shape[0], self.input_shape[1], 3))

        # Layer 1
        X = Conv2D(filters=32, kernel_size=(3, 3), strides=(2, 2), padding='same',
                   kernel_initializer=random_uniform)(X_input)

        # Block 1
        X = self.__bottleneck_block(X_input=X, filters=16, expansion_ratio=1, strides=(1, 1), is_add=False)  # Layer 2

        # Block 2
        X = self.__bottleneck_block(X_input=X, filters=24, expansion_ratio=6, strides=(2, 2), is_add=False)  # Layer 3
        X = self.__bottleneck_block(X_input=X, filters=24, expansion_ratio=6, strides=(1, 1), is_add=True)  # Layer 4

        # Block 3
        X = self.__bottleneck_block(X_input=X, filters=32, expansion_ratio=6, strides=(2, 2), is_add=False)  # Layer 5
        X = self.__bottleneck_block(X_input=X, filters=32, expansion_ratio=6, strides=(1, 1), is_add=True)  # Layer 6
        X = self.__bottleneck_block(X_input=X, filters=32, expansion_ratio=6, strides=(1, 1), is_add=True)  # Layer 7

        # Block 4
        X = self.__bottleneck_block(X_input=X, filters=64, expansion_ratio=6, strides=(2, 2), is_add=False)  # Layer 8
        X = self.__bottleneck_block(X_input=X, filters=64, expansion_ratio=6, strides=(1, 1), is_add=True)  # Layer 9
        X = self.__bottleneck_block(X_input=X, filters=64, expansion_ratio=6, strides=(1, 1), is_add=True)  # Layer 10
        X = self.__bottleneck_block(X_input=X, filters=64, expansion_ratio=6, strides=(1, 1), is_add=True)  # Layer 11

        # Block 5
        X = self.__bottleneck_block(X_input=X, filters=96, expansion_ratio=6, strides=(1, 1), is_add=False)  # Layer 12
        X = self.__bottleneck_block(X_input=X, filters=96, expansion_ratio=6, strides=(1, 1), is_add=True)  # Layer 13
        X = self.__bottleneck_block(X_input=X, filters=96, expansion_ratio=6, strides=(1, 1), is_add=True)  # Layer 14

        # Block 6
        X = self.__bottleneck_block(X_input=X, filters=160, expansion_ratio=6, strides=(2, 2), is_add=False)  # Layer 15
        X = self.__bottleneck_block(X_input=X, filters=160, expansion_ratio=6, strides=(1, 1), is_add=True)  # Layer 16
        X = self.__bottleneck_block(X_input=X, filters=160, expansion_ratio=6, strides=(1, 1), is_add=True)  # Layer 17

        # Block 7
        X = self.__bottleneck_block(X_input=X, filters=320, expansion_ratio=6, strides=(1, 1), is_add=False)  # Layer 18

        # Layer 19
        X = Conv2D(filters=1280, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu',
                   kernel_initializer=random_uniform)(X)

        # Layer 20
        X = AveragePooling2D(pool_size=(7, 7))(X)

        # Layer 21
        X = Flatten()(X)
        X = Dense(self.class_numbers, activation='softmax')(X)

        # Create model
        model = Model(inputs=X_input, outputs=X)

        return model
