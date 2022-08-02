import matplotlib.pyplot as plt
import numpy as np
import math
import cv2

from src.config import Config as Cfg


class UtilityFunction:

    @staticmethod
    def step_decay_classification(epoch):
        # Set the initial learning rate 0.1 for ResNet50, MobileNetV1, MobileNetV2, BNInception,
        # GoogLeNet, and VGG16 networks.
        # Set the initial learning rate 0.01 for InceptionV4
        initial_lr = 0.01
        drop = 0.95
        epochs_drop = 25
        lr = initial_lr * math.pow(drop, math.floor((1 + epoch) / epochs_drop))

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
        """

        fig, ax = plt.subplots()

        ax.set_ylabel(y_label)
        ax.set_xlabel(x_label)
        ax.set_title(title)

        return fig, ax
