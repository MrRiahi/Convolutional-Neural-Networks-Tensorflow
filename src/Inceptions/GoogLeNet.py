from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Activation, \
    Flatten, Dense, Input, AveragePooling2D, concatenate, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.initializers import random_uniform


class GoogLeNet:

    def __init__(self, input_shape, classes):
        self.input_shape = input_shape
        self.classes = classes

    @staticmethod
    def __naive_inception_block(X, filters, kernel_size=((1, 1), (3, 3), (5, 5)), pool_size=(3, 3),
                                initializer=random_uniform):
        """
        This method creates the naive inception block.
        :param X: input layer
        :param filters: list of filters size
        :param kernel_size: list of kernel_size
        :param pool_size: pool size of max pooling layer
        :param initializer: to set up the initial weights of a layer. Equals to random uniform initializer
        :return:
        """

        filters_1by1, filters_3by3, filters_5by5 = filters
        kernel_size_1by1, kernel_size_3by3, kernel_size_5by5 = kernel_size

        X_1by1 = Conv2D(filters=filters_1by1, kernel_size=kernel_size_1by1, strides=(1, 1),
                        padding='valid', kernel_initializer=initializer())(X)
        X_1by1 = BatchNormalization()(X_1by1)
        X_1by1 = Activation('relu')(X_1by1)

        X_3by3 = Conv2D(filters=filters_3by3, kernel_size=kernel_size_3by3, strides=(1, 1),
                        padding='valid', kernel_initializer=initializer())(X)
        X_3by3 = BatchNormalization()(X_3by3)
        X_3by3 = Activation('relu')(X_3by3)

        X_5by5 = Conv2D(filters=filters_5by5, kernel_size=kernel_size_5by5, strides=(1, 1),
                        padding='valid', kernel_initializer=initializer())(X)
        X_5by5 = BatchNormalization()(X_5by5)
        X_5by5 = Activation('relu')(X_5by5)

        X_max_pooling = MaxPooling2D(pool_size=pool_size, strides=(1, 1), padding='same')(X)

        X_concat = concatenate(inputs=[X_1by1, X_3by3, X_5by5, X_max_pooling], axis=0)
        X_concat = Activation('relu')(X_concat)

        return X_concat

    @staticmethod
    def __inception_block(X, filters, reduced_filters, kernel_size=((1, 1), (3, 3), (5, 5)), pool_size=(3, 3),
                          initializer=random_uniform):
        """
        This method creates the inception block.
        :param X: input layer
        :param filters: list of filters size
        :param reduced_filters: list of 1x1 filters for dimensionality reduction
        :param kernel_size: list of kernel_size
        :param pool_size: pool size of max pooling layer
        :param initializer: to set up the initial weights of a layer. Equals to random uniform initializer
        :return:
        """

        filters_1by1, filters_3by3, filters_5by5 = filters
        reduced_filters_3by3, reduced_filters_5by5, pool_projection = reduced_filters
        kernel_size_1by1, kernel_size_3by3, kernel_size_5by5 = kernel_size

        # 1x1 layer
        X_1by1 = Conv2D(filters=filters_1by1, kernel_size=kernel_size_1by1, strides=(1, 1),
                        padding='same', kernel_initializer=initializer())(X)
        X_1by1 = BatchNormalization()(X_1by1)
        X_1by1 = Activation('relu')(X_1by1)

        # 3x3 layer
        X_reduced_3by3 = Conv2D(filters=reduced_filters_3by3, kernel_size=(1, 1), strides=(1, 1),
                                padding='same', kernel_initializer=initializer())(X)
        X_reduced_3by3 = BatchNormalization()(X_reduced_3by3)
        X_reduced_3by3 = Activation('relu')(X_reduced_3by3)

        X_3by3 = Conv2D(filters=filters_3by3, kernel_size=kernel_size_3by3, strides=(1, 1),
                        padding='same', kernel_initializer=initializer())(X_reduced_3by3)
        X_3by3 = BatchNormalization()(X_3by3)
        X_3by3 = Activation('relu')(X_3by3)

        # 5x5 layer
        X_reduced_5by5 = Conv2D(filters=reduced_filters_5by5, kernel_size=(1, 1), strides=(1, 1),
                                padding='same', kernel_initializer=initializer())(X)
        X_reduced_5by5 = BatchNormalization()(X_reduced_5by5)
        X_reduced_5by5 = Activation('relu')(X_reduced_5by5)

        X_5by5 = Conv2D(filters=filters_5by5, kernel_size=kernel_size_5by5, strides=(1, 1),
                        padding='same', kernel_initializer=initializer())(X_reduced_5by5)
        X_5by5 = BatchNormalization()(X_5by5)
        X_5by5 = Activation('relu')(X_5by5)

        # max pooling layer
        X_max_pooling = MaxPooling2D(pool_size=pool_size, strides=(1, 1), padding='same')(X)
        X_reduced_5by5 = Conv2D(filters=pool_projection, kernel_size=(1, 1), strides=(1, 1),
                                activation='relu', padding='same',
                                kernel_initializer=initializer())(X_max_pooling)
        X_reduced_5by5 = BatchNormalization()(X_reduced_5by5)
        X_reduced_5by5 = Activation('relu')(X_reduced_5by5)

        # concatenate layers
        X_concat = concatenate(inputs=[X_1by1, X_3by3, X_5by5, X_reduced_5by5])
        X_concat = Activation('relu')(X_concat)

        return X_concat

    def __auxiliary_classifier(self, X, output_name, initializer=random_uniform):
        """
        This method creates an auxiliary classifier.
        :param X: input layer
        :param output_name: name of output layer
        :param initializer: to set up the initial weights of a layer. Equals to random uniform initializer
        :return:
        """

        # Average pooling layer
        X_average_pool = AveragePooling2D(pool_size=(5, 5), strides=(3, 3), padding='valid')(X)

        # Convolution layer for dimensionality reduction
        X_conv = Conv2D(filters=128, kernel_size=(1, 1), strides=(1, 1), padding='same',
                        kernel_initializer=initializer())(X_average_pool)
        X_conv = BatchNormalization()(X_conv)
        X_conv = Activation('relu')(X_conv)

        # Flatten layer
        X_flatten = Flatten()(X_conv)

        # FC layer
        X_fc = Dense(units=1024, activation='relu', kernel_initializer=random_uniform())(X_flatten)

        # Dropout layer
        X_dropout = Dropout(rate=0.7)(X_fc)

        # Auxiliary output layer
        X_aux_output = Dense(units=self.classes, activation='softmax', name=output_name,
                             kernel_initializer=random_uniform())(X_dropout)

        return X_aux_output

    def google_net(self):
        """
        Builds the google_net architecture
        :return:
        """

        X_input = Input((self.input_shape[0], self.input_shape[1], 3))

        # Layer 1
        X = Conv2D(filters=64, kernel_size=(7, 7), strides=(2, 2), padding='same',
                   kernel_initializer=random_uniform)(X_input)
        X = BatchNormalization()(X)
        X = Activation('relu')(X)

        # Layer 2
        X = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(X)

        # Layer 3
        X = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='valid',
                   kernel_initializer=random_uniform)(X)
        X = BatchNormalization()(X)
        X = Activation('relu')(X)

        # Layer 4
        X = Conv2D(filters=192, kernel_size=(3, 3), strides=(1, 1), padding='same',
                   kernel_initializer=random_uniform)(X)
        X = BatchNormalization()(X)
        X = Activation('relu')(X)

        # Layer 5
        X = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(X)

        # Inception 3a
        X = self.__inception_block(X=X, filters=(64, 128, 32), reduced_filters=(96, 16, 32))

        # Inception 3b
        X = self.__inception_block(X=X, filters=(128, 192, 96), reduced_filters=(128, 32, 64))

        # Layer 8
        X = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(X)

        # Inception 4a
        X = self.__inception_block(X=X, filters=(192, 208, 48), reduced_filters=(96, 16, 64))

        # First auxiliary output
        X_aux_1 = self.__auxiliary_classifier(X=X, output_name='output_aux_1')

        # Inception 4b
        X = self.__inception_block(X=X, filters=(160, 224, 64), reduced_filters=(112, 24, 64))

        # Inception 4c
        X = self.__inception_block(X=X, filters=(128, 256, 64), reduced_filters=(128, 24, 64))

        # Inception 4d
        X = self.__inception_block(X=X, filters=(112, 288, 64), reduced_filters=(144, 32, 64))

        # Second auxiliary output
        X_aux_2 = self.__auxiliary_classifier(X=X, output_name='output_aux_2')

        # Inception 4e
        X = self.__inception_block(X=X, filters=(256, 320, 128), reduced_filters=(160, 32, 128))

        # Layer 14
        X = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(X)

        # Inception 5a
        X = self.__inception_block(X=X, filters=(256, 320, 128), reduced_filters=(160, 32, 128))

        # Inception 5b
        X = self.__inception_block(X=X, filters=(384, 384, 128), reduced_filters=(192, 48, 128))

        # Layer 17
        X = AveragePooling2D(pool_size=(7, 7), strides=(1, 1), padding='valid')(X)
        X = Flatten()(X)

        # Layer 18
        X = Dropout(rate=0.4)(X)

        # Layer 19
        X = Dense(units=1000, activation='relu', kernel_initializer=random_uniform())(X)

        # Layer 20
        output = Dense(units=self.classes, activation='softmax', name='output',
                       kernel_initializer=random_uniform())(X)

        # Create model
        model = Model(inputs=X_input, outputs=[output, X_aux_1, X_aux_2])

        return model
