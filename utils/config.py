import tensorflow as tf


class Config:

    # Models config
    RESNET50_INPUT_SIZE = (64, 64)
    MOBILENET_V1_INPUT_SIZE = (224, 224)
    MODEL_TYPE = 'MobileNetV1'  # 'MobileNetV1', 'ResNet50'

    # Train config
    BATCH_SIZE = 128
    EPOCHS = 500

    # Dataset configs
    CIFAR_10_CLASS_NAMES = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog',
                            'horse', 'ship', 'truck']
    CIFAR_10_CLASS_NUMBERS = 10
