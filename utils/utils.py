import numpy as np
import math
import cv2


class UtilityFunction:

    @staticmethod
    def step_decay_classification(epoch):
        initial_lr = 0.1
        drop = 0.5
        epochs_drop = 100.0
        lr = initial_lr * math.pow(drop, math.floor((1 + epoch) / epochs_drop))
        return lr
