from skimage.transform import resize
from skimage.io import imread
import numpy as np
import math
from tensorflow.keras.utils import Sequence

from utils.config import Config as Cfg


class GoogLeNetDatasetGenerator(Sequence):

    def __init__(self, X_train_path, y_train, batch_size):
        """
        Initialize the GoogLeNet dataset generator.
        :param X_train_path: Path of train images
        :param y_train: Labels of train images
        :param batch_size:
        """

        self.X_train_path = X_train_path
        self.y_train = y_train
        self.batch_size = batch_size

        self.indexes = np.arange(len(self.X_train_path))
        np.random.shuffle(self.indexes)

    def __len__(self):
        """
        Denotes the number of batches per epoch
        :return:
        """

        return math.ceil(len(self.X_train_path) / self.batch_size)

    def __getitem__(self, index):
        """
        Get batch indexes from shuffled indexes
        :param index:
        :return:
        """

        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        X_batch_names = [self.X_train_path[i] for i in indexes]
        y_batch_naive = self.y_train[indexes]

        X_batch = np.array([resize(imread(file_name), Cfg.GOOGLE_NET_INPUT_SIZE) for file_name in X_batch_names],
                           dtype='float32')
        # y_batch = {'output': y_batch_naive, 'output_aux_1': y_batch_naive, 'output_aux_2': y_batch_naive}
        y_batch = [y_batch_naive, y_batch_naive, y_batch_naive]

        return X_batch, y_batch

    def on_epoch_end(self):
        """
        Updates indexes after each epoch
        :return:
        """

        self.indexes = np.arange(len(self.X_train_path))
        np.random.shuffle(self.indexes)
