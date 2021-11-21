import tensorflow as tf

from config import Config


def preprocess_caltech101(image, label):
    image = tf.image.resize(image, (Config.INPUT_SIZE[0], Config.INPUT_SIZE[1]))
    label = tf.one_hot(label, depth=Config.CLASSES)

    return image, label
