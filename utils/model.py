from utils.MobileNet import MobileNetV1
from utils.config import Config as Cfg
from utils.ResNet import ResNet50


def get_model():
    """
    This function builds the model defined in config.py file
    :return:
    """

    if Cfg.MODEL_TYPE == 'ResNet50':
        input_size = Cfg.RESNET50_INPUT_SIZE
        resnet50 = ResNet50(input_shape=input_size, classes=Cfg.CIFAR_10_CLASS_NUMBERS)
        model = resnet50.res_net50()

    elif Cfg.MODEL_TYPE == 'MobileNetV1':
        input_size = Cfg.MOBILENET_V1_INPUT_SIZE
        mobile_net_v1 = MobileNetV1(input_shape=input_size, classes=Cfg.CIFAR_10_CLASS_NUMBERS)
        model = mobile_net_v1.mobile_net_v1()

    else:
        raise Exception('Invalid model type')

    return model, input_size
