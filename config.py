import tensorflow as tf


class Config:

    RESNET50_INPUT_SIZE = (64, 64)
    MOBILENET_V1_INPUT_SIZE = (224, 224)

    CLASSES = 101

    AUTO = tf.data.AUTOTUNE

    BATCH_SIZE = 64
    EPOCHS = 5
