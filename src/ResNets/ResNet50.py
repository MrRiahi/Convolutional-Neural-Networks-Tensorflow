from tensorflow.keras.layers import Conv2D, MaxPooling2D,\
    Flatten, Dense, BatchNormalization, Activation, Add, Input, AveragePooling2D, ZeroPadding2D
from tensorflow.keras.models import Model
from tensorflow.keras.initializers import random_uniform, glorot_uniform


class ResNet50:
    """
    Implementation of ResNet50 network.
    """

    def __init__(self, input_shape, classes):
        self.input_shape = input_shape
        self.classes = classes

    @staticmethod
    def __identity_block(X, f, filters, initializer=random_uniform):
        """
        Implementation of the identity block in ResNet.
        :param X: input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)
        :param f: shape of the middle conv window for the main path
        :param filters: number of filters in conv layers in the main path
        :param initializer: to set up the initial weights of a layer. Equals to random uniform initializer
        :return: X output of the identity block which is a tensor of shape (m, n_H, n_W, n_C)
        """

        # Retrieve Filters
        filters_1, filters_2, filters_3 = filters

        # Save the input value. You'll need this later to add back to the main path.
        X_shortcut = X

        # First component of main path
        X = Conv2D(filters=filters_1, kernel_size=(1, 1), strides=(1, 1), padding='valid',
                   kernel_initializer=initializer(seed=0))(X)
        X = BatchNormalization()(X)
        X = Activation('relu')(X)

        # Second component of main path
        X = Conv2D(filters=filters_2, kernel_size=(f, f), strides=(1, 1), padding='same',
                   kernel_initializer=initializer(seed=0))(X)
        X = BatchNormalization()(X)
        X = Activation('relu')(X)

        # Third component of main path
        X = Conv2D(filters=filters_3, kernel_size=(1, 1), strides=(1, 1), padding='valid',
                   kernel_initializer=initializer(seed=0))(X)
        X = BatchNormalization()(X)
        X = Activation('relu')(X)

        # Final step: Add shortcut value to main path, and pass it through a RELU activation (≈2 lines)
        X = Add()([X_shortcut, X])
        X = Activation('relu')(X)

        return X

    @staticmethod
    def __convolutional_block(X, f, filters, stride=2, initializer=glorot_uniform):
        """
        Implementation of the convolutional block of ResNet
        :param X: input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)
        :param f: shape of the middle conv window for the main path
        :param filters: number of filters in conv layers in the main path
        :param stride: shape of stride in conv layers
        :param initializer: o set up the initial weights of a layer. Default is Glorot uniform initializer
        :return: X output of the convolutional block which is a tensor of shape (m, n_H, n_W, n_C)
        """

        # Retrieve Filters
        filters_1, filters_2, filters_3 = filters

        # Save the input value
        X_shortcut = X

        # Main path
        # First component of main path
        X = Conv2D(filters=filters_1, kernel_size=(1, 1), strides=(stride, stride), padding='valid',
                   kernel_initializer=initializer(seed=0))(X)
        X = BatchNormalization()(X)
        X = Activation('relu')(X)

        # Second component of main path
        X = Conv2D(filters=filters_2, kernel_size=(f, f), strides=(1, 1), padding='same',
                   kernel_initializer=initializer(seed=0))(X)
        X = BatchNormalization()(X)
        X = Activation('relu')(X)

        # Third component of main path
        X = Conv2D(filters=filters_3, kernel_size=(1, 1), strides=(1, 1), padding='valid',
                   kernel_initializer=initializer(seed=0))(X)
        X = BatchNormalization()(X)
        X = Activation('relu')(X)

        # Shortcut path
        X_shortcut = Conv2D(filters=filters_3, kernel_size=(1, 1), strides=(stride, stride), padding='valid',
                            kernel_initializer=initializer(seed=0))(X_shortcut)
        X_shortcut = BatchNormalization()(X_shortcut)
        X_shortcut = Activation('relu')(X_shortcut)

        # Add shortcut path to main path
        X = Add()([X, X_shortcut])
        X = Activation('relu')(X)

        return X

    def __call__(self):
        """
        Build the ResNe50 model
        :return: ResNet50 model
        """

        # Define the input as a tensor with shape input_shape
        X_input = Input((self.input_shape[0], self.input_shape[1], 3))

        # Zero-Padding
        X = ZeroPadding2D((3, 3))(X_input)

        # Stage 1
        X = Conv2D(64, (7, 7), strides=(2, 2), kernel_initializer=glorot_uniform(seed=0))(X)
        X = BatchNormalization(axis=3)(X)
        X = Activation('relu')(X)

        X = MaxPooling2D((3, 3), strides=(2, 2))(X)

        # Stage 2
        X = self.__convolutional_block(X, f=3, filters=[64, 64, 256], stride=1)
        X = self.__identity_block(X, 3, [64, 64, 256])
        X = self.__identity_block(X, 3, [64, 64, 256])

        # Stage 3
        X = self.__convolutional_block(X=X, f=3, filters=[128, 128, 512], stride=2)
        X = self.__identity_block(X=X, f=3, filters=[128, 128, 512])
        X = self.__identity_block(X=X, f=3, filters=[128, 128, 512])
        X = self.__identity_block(X=X, f=3, filters=[128, 128, 512])

        # Stage 4
        X = self.__convolutional_block(X=X, f=3, filters=[256, 256, 1024], stride=2)
        X = self.__identity_block(X=X, f=3, filters=[256, 256, 1024])
        X = self.__identity_block(X=X, f=3, filters=[256, 256, 1024])
        X = self.__identity_block(X=X, f=3, filters=[256, 256, 1024])
        X = self.__identity_block(X=X, f=3, filters=[256, 256, 1024])
        X = self.__identity_block(X=X, f=3, filters=[256, 256, 1024])

        # Stage 5
        X = self.__convolutional_block(X=X, f=3, filters=[512, 512, 2048], stride=2)
        X = self.__identity_block(X=X, f=3, filters=[512, 512, 2048])
        X = self.__identity_block(X=X, f=3, filters=[512, 512, 2048])

        # Average pooling
        X = AveragePooling2D()(X)

        # Output layer
        X = Flatten()(X)
        X = Dense(self.classes, activation='softmax', kernel_initializer=glorot_uniform(seed=0))(X)

        # Create model
        model = Model(inputs=X_input, outputs=X)

        return model
