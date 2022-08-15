from tensorflow.keras.layers import Conv2D, BatchNormalization, MaxPooling2D, Activation, \
    Input, Dense, Flatten, Dropout, AveragePooling2D
from tensorflow.keras.initializers import glorot_uniform
from tensorflow.keras.models import Model


class VGG16:
    """
    Implementation of VGG16 network.
    """

    def __init__(self, input_shape, classes):
        """
        Initialize the VGG16 network.
        :param input_shape: input shape of image
        :param classes: number of classes
        :return:
        """
        self.input_shape = input_shape
        self.classes = classes

    @staticmethod
    def __conv_block(X, number_of_filters, kernel_size=3, stride=1):
        """
        Define convolution block.
        :param X: input layer of the conv_block
        :param number_of_filters: a list that contains the number of filters for each convolution layer.
        :param kernel_size: kernel size of the convolution layers.
        :param stride: stride of the convolution layers.
        :return
        """

        for i_filters in number_of_filters:
            X = Conv2D(filters=i_filters, kernel_size=(kernel_size, kernel_size),
                       strides=(stride, stride), padding='same',
                       kernel_initializer=glorot_uniform())(X)

            X = BatchNormalization()(X)
            X = Activation('relu')(X)

        return X

    def vgg16(self):
        """
        Build the VGG16 network.
        :return: vgg16 model
        """

        # Define the input as a tensor with shape input_shape
        X_input = Input((self.input_shape[0], self.input_shape[1], 3))

        # First conv block
        X = self.__conv_block(X=X_input, number_of_filters=[64, 64])

        # Max pooling
        X = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(X)

        # Second conv block
        X = self.__conv_block(X=X, number_of_filters=[128, 128])

        # Max pooling
        X = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(X)

        # Third conv block
        X = self.__conv_block(X=X, number_of_filters=[256, 256, 256])

        # Max pooling
        X = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(X)

        # Forth conv block
        X = self.__conv_block(X=X, number_of_filters=[512, 512, 512])

        # Max pooling
        X = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(X)

        # Fifth conv block
        X = self.__conv_block(X=X, number_of_filters=[512, 512, 512])

        # In original VGG, a max pooling was used. Because of limited resources, I use average pooling with
        # 7x7 pool size.
        # Max pooling
        X = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(X)

        # # Average pooling
        # X = AveragePooling2D(pool_size=[7, 7])(X)

        # FC layers
        X = Flatten()(X)
        X = Dense(units=4096, activation='relu', kernel_initializer=glorot_uniform())(X)
        X = Dropout(rate=0.5)(X)

        X = Dense(units=4096, activation='relu', kernel_initializer=glorot_uniform())(X)
        X = Dropout(rate=0.5)(X)

        X_output = Dense(units=self.classes, activation='softmax', kernel_initializer=glorot_uniform())(X)

        model = Model(inputs=X_input, outputs=X_output)

        return model
