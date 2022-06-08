
class Config:

    # Models config
    RESNET50_INPUT_SIZE = (64, 64)
    MOBILENET_V1_INPUT_SIZE = (224, 224)
    MOBILENET_V2_INPUT_SIZE = (224, 224)
    GOOGLE_NET_INPUT_SIZE = (224, 224)

    MODEL_TYPE = 'GoogLeNet'  # 'MobileNetV1', 'MobileNetV2', 'ResNet50', 'GoogLeNet'

    # Train config
    BUFFER_SIZE = 500
    BATCH_SIZE = 128
    EPOCHS = 500

    TRAIN_SUBSET = 0.8
    VALIDATION_SUBSET = 1 - TRAIN_SUBSET

    TRAIN_DATASET_PATH = 'dataset/cifar-10/images/train'
    TEST_DATASET_PATH = 'dataset/cifar-10/images/test'

    # Dataset configs
    CIFAR_10_CLASS_NAMES = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog',
                            'horse', 'ship', 'truck']
    CIFAR10_CLASS_NAME_TO_NUMBER = {'airplane': 0, 'automobile': 1, 'bird': 2, 'cat': 3,
                                    'deer': 4, 'dog': 5, 'frog': 6, 'horse': 7, 'ship': 8, 'truck': 9}
    CIFAR_10_CLASS_NUMBERS = 10
