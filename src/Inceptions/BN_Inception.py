from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Activation, \
    Flatten, Dense, Input, AveragePooling2D, concatenate, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.initializers import random_uniform


class BNInception:
    """
    Implementation of Inception version 2.
    """

    def __init__(self, input_shape, classes):
        self.input_shape = input_shape
        self.classes = classes

    @staticmethod
    def __1by1_block(X, filters_1by1, strides, initializer=random_uniform):
        """
        Implementation of the 3by3 block. This block consist of 1x1 and 3x3 conv layers.
        :param X: input layer
        :param filters_1by1: number of 1x1 filters
        :param strides: size of stride
        :param initializer: to set up the initial weights of a layer. Equals to random uniform initializer
        :return:
        """

        X_1by1 = None

        if filters_1by1 != 0:
            X_1by1 = Conv2D(filters=filters_1by1, kernel_size=strides, strides=(1, 1),
                            padding='same', kernel_initializer=initializer())(X)
            X_1by1 = BatchNormalization()(X_1by1)
            X_1by1 = Activation('relu')(X_1by1)

        return X_1by1

    @staticmethod
    def __3by3_block(X, reduced_filters_3by3, filters_3by3, strides, initializer=random_uniform):
        """
        Implementation of the 1by1 block. This block consist of 1x1 layer.
        :param X: input layer
        :param reduced_filters_3by3: number of 1x1 filters for dimensionality reduction
        :param filters_3by3: number of 3x3 filters
        :param strides: size of stride
        :param initializer: to set up the initial weights of a layer. Equals to random uniform initializer
        :return:
        """

        X_reduced_3by3 = Conv2D(filters=reduced_filters_3by3, kernel_size=(1, 1), strides=(1, 1),
                                padding='same', kernel_initializer=initializer())(X)
        X_reduced_3by3 = BatchNormalization()(X_reduced_3by3)
        X_reduced_3by3 = Activation('relu')(X_reduced_3by3)

        X_3by3 = Conv2D(filters=filters_3by3, kernel_size=(3, 3), strides=strides,
                        padding='same', kernel_initializer=initializer())(X_reduced_3by3)
        X_3by3 = BatchNormalization()(X_3by3)
        X_3by3 = Activation('relu')(X_3by3)

        return X_3by3

    @staticmethod
    def __3by3_double_block(X, double_reduced_filters_3by3, double_filters_3by3,
                            strides, initializer=random_uniform):
        """
        Implementation of the double 3by3 block. This block consist of 1x1 and double 3x3 conv layers.
        :param X: input layer
        :param double_reduced_filters_3by3: number of 1x1 filters for dimensionality reduction
        :param double_filters_3by3: number of 3x3 filters for double 3x3 layers
        :param strides: size of stride.
        :param initializer: to set up the initial weights of a layer. Equals to random uniform initializer
        :return:
        """

        # 1x1 layer
        X_double_reduced_3by3 = Conv2D(filters=double_reduced_filters_3by3, kernel_size=(1, 1),
                                       strides=(1, 1), padding='same', kernel_initializer=initializer())(X)
        X_double_reduced_3by3 = BatchNormalization()(X_double_reduced_3by3)
        X_double_reduced_3by3 = Activation('relu')(X_double_reduced_3by3)

        # first 3x3 layer
        X_double_3by_3 = Conv2D(filters=double_filters_3by3, kernel_size=(3, 3), strides=(1, 1),
                                padding='same', kernel_initializer=initializer())(X_double_reduced_3by3)
        X_double_3by_3 = BatchNormalization()(X_double_3by_3)
        X_double_3by_3 = Activation('relu')(X_double_3by_3)

        # second 3x3 layer
        X_double_3by_3 = Conv2D(filters=double_filters_3by3, kernel_size=(3, 3), strides=strides,
                                padding='same', kernel_initializer=initializer())(X_double_3by_3)
        X_double_3by_3 = BatchNormalization()(X_double_3by_3)
        X_double_3by_3 = Activation('relu')(X_double_3by_3)

        return X_double_3by_3

    @staticmethod
    def __pooling_block(X, pool_projection, pool_type, strides, initializer=random_uniform):
        """
        Implementation of the pooling block. This block consist of pooling and 1x1 layer.
        :param X: input layer
        :param pool_projection: number of 1x1 filters for dimensionality reduction
        :param pool_type: type of pooling layer. max pooling or average pooling
        :param strides: size of stride.
        :param initializer: to set up the initial weights of a layer. Equals to random uniform initializer
        :return:
        """

        X_pooling = None

        if pool_type == 'max':
            X_pooling = MaxPooling2D(pool_size=(3, 3), strides=strides, padding='same')(X)
        elif pool_type == 'avg':
            X_pooling = AveragePooling2D(pool_size=(3, 3), strides=strides, padding='same')(X)

        if pool_projection != 0:
            X_pooling = Conv2D(filters=pool_projection, kernel_size=(1, 1), strides=strides,
                               activation='relu', padding='same',
                               kernel_initializer=initializer())(X_pooling)

            X_pooling = BatchNormalization()(X_pooling)
            X_pooling = Activation('relu')(X_pooling)

        return X_pooling

    def __inception_v2_block(self, X, filters, reduced_filters, pool_type='avg', strides=(1, 1),
                             initializer=random_uniform):
        """
        This method creates the inception block.
        :param X: input layer
        :param filters: list of the number of filters
        :param reduced_filters: list of the number of 1x1 filters for dimensionality reduction
        :param pool_type: type of pooling layer. max pooling or average pooling
        :param strides: size of stride. For some layers are (2, 2) while for the others are (1, 1)
        :param initializer: to set up the initial weights of a layer. Equals to random uniform initializer
        :return:
        """

        filters_1by1, filters_3by3, double_filters_3by3 = filters
        reduced_filters_3by3, double_reduced_filters_3by3, pool_projection = reduced_filters

        # 1x1 layer
        X_1by1 = self.__1by1_block(X=X, filters_1by1=filters_1by1, strides=strides,
                                   initializer=initializer)

        # 3x3 layer
        X_3by3 = self.__3by3_block(X=X, reduced_filters_3by3=reduced_filters_3by3,
                                   filters_3by3=filters_3by3,
                                   strides=strides, initializer=initializer)

        # Double 3x3 layers
        X_double_3by_3 = self.__3by3_double_block(X=X,
                                                  double_reduced_filters_3by3=double_reduced_filters_3by3,
                                                  double_filters_3by3=double_filters_3by3,
                                                  strides=strides,
                                                  initializer=initializer)

        # max pooling layer
        X_pooling = self.__pooling_block(X=X, pool_projection=pool_projection,
                                         pool_type=pool_type, strides=strides,
                                         initializer=random_uniform)

        # concatenate layers
        if X_1by1 is None:
            X_concat = concatenate(inputs=[X_3by3, X_double_3by_3, X_pooling])
        else:
            X_concat = concatenate(inputs=[X_1by1, X_3by3, X_double_3by_3, X_pooling])

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

    def inception_bn(self):
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
        X = self.__inception_v2_block(X=X, filters=(64, 64, 96), reduced_filters=(64, 64, 32),
                                      pool_type='avg')

        # Inception 3b
        X = self.__inception_v2_block(X=X, filters=(64, 96, 96), reduced_filters=(64, 64, 64),
                                      pool_type='avg')

        # Inception 3c
        X = self.__inception_v2_block(X=X, filters=(0, 160, 96), reduced_filters=(64, 64, 0),
                                      strides=(2, 2), pool_type='max')

        # Inception 4a
        X = self.__inception_v2_block(X=X, filters=(224, 96, 128), reduced_filters=(64, 96, 128),
                                      pool_type='avg')

        # Inception 4b
        X = self.__inception_v2_block(X=X, filters=(192, 128, 128), reduced_filters=(96, 96, 128),
                                      pool_type='avg')

        # Inception 4c
        X = self.__inception_v2_block(X=X, filters=(160, 160, 160), reduced_filters=(128, 128, 128),
                                      pool_type='avg')

        # Inception 4d
        X = self.__inception_v2_block(X=X, filters=(96, 192, 192), reduced_filters=(128, 160, 128),
                                      pool_type='avg')

        # Inception 4e
        X = self.__inception_v2_block(X=X, filters=(0, 192, 256), reduced_filters=(128, 192, 0),
                                      strides=(2, 2), pool_type='max')

        # Inception 5a
        X = self.__inception_v2_block(X=X, filters=(352, 320, 224), reduced_filters=(192, 160, 128),
                                      pool_type='avg')

        # Inception 5b
        X = self.__inception_v2_block(X=X, filters=(352, 320, 224), reduced_filters=(192, 192, 128),
                                      pool_type='max')

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
        model = Model(inputs=X_input, outputs=output)

        return model
