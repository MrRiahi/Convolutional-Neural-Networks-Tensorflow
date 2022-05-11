import numpy as np
import math
import cv2

from utils.config import Config as Cfg


class UtilityFunction:

    @staticmethod
    def step_decay_classification(epoch):
        initial_lr = 0.1
        drop = 0.5
        epochs_drop = 100.0
        lr = initial_lr * math.pow(drop, math.floor((1 + epoch) / epochs_drop))
        return lr

    @staticmethod
    def load_images(images_path):
        """
        Load images from images_path directory.
        :param images_path: images directory
        """

        images = [cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB) / 255
                  for image_path in images_path]

        images = [cv2.resize(image, dsize=Cfg.RESNET50_INPUT_SIZE) for image in images]

        return np.array(images)

    @staticmethod
    def get_predictions_label(predictions):
        """
        Get labels of the predictions.
        :param predictions: list of predictions
        """

        labels = [Cfg.CIFAR_10_CLASS_NAMES[np.argmax(prediction)] for prediction in predictions]

        return labels
