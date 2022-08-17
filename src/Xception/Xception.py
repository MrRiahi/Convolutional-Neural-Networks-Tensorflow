from tensorflow.keras.layers import Conv2D, BatchNormalization, DepthwiseConv2D, Activation, Input, \
                                    Dense, Flatten, Add, MaxPooling2D, GlobalAveragePooling2D
from tensorflow.keras.initializers import random_uniform
from tensorflow.keras.models import Model


class Xception:
    """
    This class is the implementation of the Xception convolutional neural network.
    """

    def __init__(self, input_shape, classes):
        """
        Initialization of Xception class
        :param input_shape:
        :param classes:
        """

        self.input_shape = input_shape
        self.class_numbers = classes

    @staticmethod
    def __separable_conv2d(X, filters):
        """
        Implementation of the separable convolution block.
        :param X:
        :param filters:
        :return:
        """

        # Depth wise part
        X = DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same',
                            depthwise_initializer=random_uniform)(X)
        X = BatchNormalization()(X)

        # Point wise part
        X = Conv2D(filters=filters, kernel_size=(1, 1), strides=(1, 1), padding='same',
                   kernel_initializer=random_uniform)(X)
        X = BatchNormalization()(X)

        return X

    def __separable_block(self, X, filters1, filters2, is_entry=True):
        """
        Implementation of the separable block. This block consist of two parallel branches.
        first branch: separable_conv2d --> ReLu --> separable_conv2d --> MaxPooling2d
        second branch: conv2d 1x1
        final branch is the sum of the first and second branches.
        :param X:
        :param filters1: number of filters for the first separable convolution block
        :param filters2: number of filters for the second separable convolution block
        :param is_entry: if it is True, use Conv2D 1x1 and MaxPooling2D
        :return:
        """

        X_b1 = Activation('relu')(X)
        X_b1 = self.__separable_conv2d(X=X_b1, filters=filters1)

        X_b1 = Activation('relu')(X_b1)
        X_b1 = self.__separable_conv2d(X=X_b1, filters=filters2)

        if is_entry:
            X_b1 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(X_b1)

            X_b2 = Conv2D(filters=filters2, kernel_size=(1, 1), strides=(2, 2), padding='same',
                          kernel_initializer=random_uniform)(X)

        else:
            X_b2 = X

        X = Add()([X_b1, X_b2])

        return X

    def __entry_flow(self, X):
        """
        Implementation of the entry flow part of the Xception model.
        :param X:
        :return:
        """

        # first layer
        X = Conv2D(filters=32, kernel_size=(3, 3), strides=(2, 2), padding='valid',
                   kernel_initializer=random_uniform)(X)
        X = BatchNormalization()(X)
        X = Activation('relu')(X)

        # second layer
        X = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same',
                   kernel_initializer=random_uniform)(X)
        X = BatchNormalization()(X)
        X = Activation('relu')(X)

        # third block
        X = self.__separable_block(X=X, filters1=128, filters2=128)

        # forth block
        X = self.__separable_block(X=X, filters1=256, filters2=256)

        # fifth block
        X = self.__separable_block(X=X, filters1=728, filters2=728)

        return X

    def __middle_flow(self, X):
        """
        Implementation of the middle flow part of the Xception model.
        :param X:
        :return:
        """

        # first block
        X_b1 = Activation('relu')(X)
        X_b1 = self.__separable_block(X=X_b1, filters1=728, filters2=728, is_entry=False)

        # second block
        X_b1 = Activation('relu')(X_b1)
        X_b1 = self.__separable_block(X=X_b1, filters1=728, filters2=728, is_entry=False)

        # third block
        X_b1 = Activation('relu')(X_b1)
        X_b1 = self.__separable_block(X=X_b1, filters1=728, filters2=728, is_entry=False)

        # add branches
        X = Add()([X, X_b1])

        return X

    def __exit_flow(self, X):
        """
        Implementation of the exit flow part of the Xception model.
        :param X:
        :return:
        """

        # first block
        X = self.__separable_block(X=X, filters1=728, filters2=1024)

        # second block
        X = self.__separable_conv2d(X=X, filters=1536)
        X = Activation('relu')(X)

        # third block
        X = self.__separable_conv2d(X=X, filters=2048)
        X = Activation('relu')(X)

        return X

    def __call__(self):

        X_input = Input(shape=(self.input_shape[0], self.input_shape[1], 3))

        # entry part
        X = self.__entry_flow(X=X_input)

        # middle part
        for i in range(8):
            X = self.__middle_flow(X=X)

        # exit part
        X = self.__exit_flow(X=X)

        # classification part
        X = GlobalAveragePooling2D()(X)

        X = Flatten()(X)
        X = Dense(units=2048, activation='relu', kernel_initializer=random_uniform)(X)

        X_output = Dense(units=self.class_numbers, activation='softmax', kernel_initializer=random_uniform)(X)

        model = Model(inputs=X_input, outputs=X_output)

        return model
