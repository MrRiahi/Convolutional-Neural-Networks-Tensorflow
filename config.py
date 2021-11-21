import tensorflow as tf


class Config:

    INPUT_SIZE = (64, 64)
    CLASSES = 101

    AUTO = tf.data.AUTOTUNE

    BATCH_SIZE = 64
    EPOCHS = 5
