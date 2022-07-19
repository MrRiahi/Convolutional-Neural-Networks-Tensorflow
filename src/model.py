from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.optimizers import SGD
import tensorflow as tf

from src.MobileNets.MobileNetV1 import MobileNetV1
from src.MobileNets.MobileNetV2 import MobileNetV2
from src.VGGs.VGG16 import VGG16
from src.VGGs.VGG13 import VGG13
from src.VGGs.VGG11 import VGG11
from src.Inceptions.GoogLeNet import GoogLeNet
from src.Inceptions.BN_Inception import BNInception
from src.ResNets.ResNet50 import ResNet50

from src.config import Config as Cfg


def get_model(classes_numbers):
    """
    This function builds the model defined in config.py file
    :param classes_numbers:
    :return:
    """

    if Cfg.MODEL_TYPE == 'ResNet50':
        # Build model
        input_size = Cfg.RESNET50_INPUT_SIZE
        resnet50 = ResNet50(input_shape=input_size, classes=classes_numbers)
        model = resnet50.res_net50()

        # Compile model
        model.compile(loss=CategoricalCrossentropy(), optimizer='adam', metrics=['accuracy'])

    elif Cfg.MODEL_TYPE == 'MobileNetV1':
        # Build model
        input_size = Cfg.MOBILENET_V1_INPUT_SIZE
        mobile_net_v1 = MobileNetV1(input_shape=input_size, classes=classes_numbers)
        model = mobile_net_v1.mobile_net_v1()

        # Compile model
        model.compile(loss=CategoricalCrossentropy(), optimizer='adam', metrics=['accuracy'])

    elif Cfg.MODEL_TYPE == 'MobileNetV2':
        # Build model
        input_size = Cfg.MOBILENET_V2_INPUT_SIZE
        mobile_net_v2 = MobileNetV2(input_shape=input_size, classes=classes_numbers)
        model = mobile_net_v2.mobile_net_v2()

        # Compile model
        model.compile(loss=CategoricalCrossentropy(), optimizer='sgd', metrics=['accuracy'])

    elif Cfg.MODEL_TYPE == 'GoogLeNet':
        # Build model
        input_size = Cfg.GOOGLE_NET_INPUT_SIZE
        google_net = GoogLeNet(input_shape=input_size, classes=classes_numbers)
        model = google_net.google_net()

        # Compile model
        losses = {'output': CategoricalCrossentropy(), 'output_aux_1': CategoricalCrossentropy(),
                  'output_aux_2': CategoricalCrossentropy()}

        metrics = {'output': 'accuracy', 'output_aux_1': 'accuracy', 'output_aux_2': 'accuracy'}

        optimizer = SGD(learning_rate=0.1, momentum=0.9)

        model.compile(loss=losses,
                      optimizer=optimizer,
                      loss_weights=[1, 0.3, 0.3],
                      metrics=metrics)

    elif Cfg.MODEL_TYPE == 'VGG16':
        # Build model
        input_size = Cfg.VGG16_NET_INPUT_SIZE
        vgg16_net = VGG16(input_shape=input_size, classes=classes_numbers)
        model = vgg16_net.vgg16()

        # Compile model
        optimizer = SGD(learning_rate=0.1, momentum=0.9)

        model.compile(loss=CategoricalCrossentropy(), optimizer=optimizer, metrics=['accuracy'])

    elif Cfg.MODEL_TYPE == 'VGG13':
        # Build model
        input_size = Cfg.VGG13_NET_INPUT_SIZE
        vgg13_net = VGG13(input_shape=input_size, classes=classes_numbers)
        model = vgg13_net.vgg13()

        # Compile model
        model.compile(loss=CategoricalCrossentropy(), optimizer='adam', metrics=['accuracy'])

    elif Cfg.MODEL_TYPE == 'VGG11':
        # Build model
        input_size = Cfg.VGG11_NET_INPUT_SIZE
        vgg11_net = VGG11(input_shape=input_size, classes=classes_numbers)
        model = vgg11_net.vgg11()

        # Compile model
        model.compile(loss=CategoricalCrossentropy(), optimizer='adam', metrics=['accuracy'])

    elif Cfg.MODEL_TYPE == 'BNInception':
        # Build model
        input_size = Cfg.INCEPTION_BN_INPUT_SIZE
        inception_bn_net = BNInception(input_shape=input_size, classes=classes_numbers)
        model = inception_bn_net.inception_bn()

        # Compile model
        optimizer = SGD(learning_rate=0.1, momentum=0.9)

        model.compile(loss=CategoricalCrossentropy(), optimizer=optimizer, metrics=['accuracy'])

    else:
        raise Exception('Invalid model type!')

    return model, input_size


def load_model(model_path):
    """
    This function loads the model in model_path
    :param model_path:
    :return:
    """

    if Cfg.MODEL_TYPE == 'ResNet50':
        input_shape = Cfg.RESNET50_INPUT_SIZE

        # Load model
        model = tf.keras.models.load_model(model_path)

    elif Cfg.MODEL_TYPE == 'MobileNetV1':
        input_shape = Cfg.MOBILENET_V1_INPUT_SIZE

        # Load model
        model = tf.keras.models.load_model(model_path)

    elif Cfg.MODEL_TYPE == 'MobileNetV2':
        input_shape = Cfg.MOBILENET_V2_INPUT_SIZE

        # Load model
        model = tf.keras.models.load_model(model_path)

    elif Cfg.MODEL_TYPE == 'GoogLeNet':
        input_shape = Cfg.GOOGLE_NET_INPUT_SIZE

        # Load model
        model = tf.keras.models.load_model(model_path)

    elif Cfg.MODEL_TYPE == 'VGG16':
        input_shape = Cfg.VGG16_NET_INPUT_SIZE

        # Load model
        model = tf.keras.models.load_model(model_path)

    elif Cfg.MODEL_TYPE == 'VGG13':
        input_shape = Cfg.VGG13_NET_INPUT_SIZE

        # Load model
        model = tf.keras.models.load_model(model_path)

    elif Cfg.MODEL_TYPE == 'VGG11':
        input_shape = Cfg.VGG11_NET_INPUT_SIZE

        # Load model
        model = tf.keras.models.load_model(model_path)

    elif Cfg.MODEL_TYPE == 'BNInception':
        input_shape = Cfg.INCEPTION_BN_INPUT_SIZE

        # Load model
        model = tf.keras.models.load_model(model_path)

    elif Cfg.MODEL_TYPE == 'InceptionV3':
        input_shape = Cfg.INCEPTION_V3_INPUT_SIZE

        # Load model
        model = tf.keras.models.load_model(model_path)

    elif Cfg.MODEL_TYPE == 'Xception':
        input_shape = Cfg.XCEPTION_INPUT_SIZE

        # Load model
        model = tf.keras.models.load_model(model_path)

    else:
        raise Exception('Invalid model type!')

    return model, input_shape
