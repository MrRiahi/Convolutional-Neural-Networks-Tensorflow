import matplotlib.pyplot as plt
import numpy as np
import math
import cv2
import os

from src.config import Config as Cfg


class UtilityFunction:

    @staticmethod
    def learning_rate_decay(epoch, lr):
        """
        The method use step decay for learning rate based on the epochs. If the fine_tune flag is False,
        it uses a predefined value for learning rate. Otherwise, it starts the initial learning rate from the save
        history file.
        :param epoch:
        :param lr:
        :return:
        """

        drop = 0.97
        epochs_drop = 25
        lr = lr * math.pow(drop, math.floor((1 + epoch) / epochs_drop))

        return lr

    @staticmethod
    def load_images(images_path, input_shape):
        """
        Load images from images_path directory.
        :param images_path: images directory
        :param input_shape: model input shape
        """

        images = [cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB) / 255
                  for image_path in images_path]

        images = [cv2.resize(image, dsize=input_shape) for image in images]

        return np.array(images)

    @staticmethod
    def get_predictions_label(predictions):
        """
        Get labels of the predictions.
        :param predictions: list of predictions
        :return:
        """

        labels = [Cfg.CIFAR_10_CLASS_NAMES[np.argmax(prediction)] for prediction in predictions]

        return labels

    @staticmethod
    def create_figure(title, x_label, y_label):
        """
        Create a figure with title, x_label, and y_label.
        :param title: figure title
        :param x_label: label of x axis
        :param y_label: label of y axis
        :return:
        """

        fig, ax = plt.subplots()

        ax.set_ylabel(y_label)
        ax.set_xlabel(x_label)
        ax.set_title(title)

        return fig, ax

    @ staticmethod
    def save_history(history):
        """
        Save the history of the trained model. This history contains the learning_rate (as lr), training loss (as loss),
        training accuracy (as accuracy), validation loss (as val_loss), and validation accuracy (as val_accuracy).
        If there is any previous history, the current history is appended to the end of the previous history
        and will be saved.
        :param history:
        :return:
        """

        history_path = f'{Cfg.MODEL_PATH}/history.npy'

        if os.path.exists(path=history_path):
            # Append the history to the previous history.
            previous_history = np.load(history_path, allow_pickle=True)
            previous_history = previous_history.tolist()

            new_history = {}
            for key, prev_value in previous_history.items():
                new_history[key] = prev_value
                new_history[key].extend(history[key])

            history = new_history

        # Sve the history.
        np.save(history_path, history)
