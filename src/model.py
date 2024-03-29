from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.optimizers import SGD, RMSprop, Adam
import tensorflow as tf

from src.Inception_ResNets.Inception_ResNetV1 import InceptionResNetV1
from src.Inception_ResNets.Inception_ResNetV2 import InceptionResNetV2
from src.Inceptions.BN_Inception import BNInception
from src.MobileNets.MobileNetV1 import MobileNetV1
from src.MobileNets.MobileNetV2 import MobileNetV2
from src.Inceptions.InceptionV3 import InceptionV3
from src.Inceptions.InceptionV4 import InceptionV4
from src.Inceptions.GoogLeNet import GoogLeNet
from src.Xception.Xception import Xception
from src.ResNets.ResNet50 import ResNet50
from src.VGGs.VGG16 import VGG16
from src.VGGs.VGG13 import VGG13
from src.VGGs.VGG11 import VGG11

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
        model = resnet50()

        # Compile model
        optimizer = Adam(learning_rate=0.1)

        model.compile(loss=CategoricalCrossentropy(), optimizer=optimizer, metrics=['accuracy'])

    elif Cfg.MODEL_TYPE == 'MobileNetV1':
        # Build model
        input_size = Cfg.MOBILENET_V1_INPUT_SIZE

        mobileV1_obj = MobileNetV1(input_shape=input_size, classes=classes_numbers)
        model = mobileV1_obj()

        # Compile model
        optimizer = Adam(learning_rate=0.1)

        model.compile(loss=CategoricalCrossentropy(), optimizer=optimizer, metrics=['accuracy'])

    elif Cfg.MODEL_TYPE == 'MobileNetV2':
        # Build model
        input_size = Cfg.MOBILENET_V2_INPUT_SIZE

        mobileV2_obj = MobileNetV2(input_shape=input_size, classes=classes_numbers)
        model = mobileV2_obj()

        # Compile model
        optimizer = SGD(learning_rate=0.1)

        model.compile(loss=CategoricalCrossentropy(), optimizer=optimizer, metrics=['accuracy'])

    elif Cfg.MODEL_TYPE == 'GoogLeNet':
        # Build model
        input_size = Cfg.GOOGLE_NET_INPUT_SIZE

        google_obj = GoogLeNet(input_shape=input_size, classes=classes_numbers)
        model = google_obj()

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

        vgg16_obj = VGG16(input_shape=input_size, classes=classes_numbers)
        model = vgg16_obj()

        # Compile model
        optimizer = SGD(learning_rate=0.1, momentum=0.9)

        model.compile(loss=CategoricalCrossentropy(), optimizer=optimizer, metrics=['accuracy'])

    elif Cfg.MODEL_TYPE == 'VGG13':
        # Build model
        input_size = Cfg.VGG13_NET_INPUT_SIZE

        vgg13_obj = VGG13(input_shape=input_size, classes=classes_numbers)
        model = vgg13_obj()

        # Compile model
        optimizer = SGD(learning_rate=0.1, momentum=0.9)

        model.compile(loss=CategoricalCrossentropy(), optimizer=optimizer, metrics=['accuracy'])

    elif Cfg.MODEL_TYPE == 'VGG11':
        # Build model
        input_size = Cfg.VGG11_NET_INPUT_SIZE

        vgg11_obj = VGG11(input_shape=input_size, classes=classes_numbers)
        model = vgg11_obj()

        # Compile model
        optimizer = SGD(learning_rate=0.1)

        model.compile(loss=CategoricalCrossentropy(), optimizer=optimizer, metrics=['accuracy'])

    elif Cfg.MODEL_TYPE == 'BNInception':
        # Build model
        input_size = Cfg.INCEPTION_BN_INPUT_SIZE

        inception_bn_obj = BNInception(input_shape=input_size, classes=classes_numbers)
        model = inception_bn_obj()

        # Optimizer
        optimizer = SGD(learning_rate=0.1, momentum=0.9)

        # Compile model
        model.compile(loss=CategoricalCrossentropy(), optimizer=optimizer, metrics=['accuracy'])

    elif Cfg.MODEL_TYPE == 'InceptionV3':
        pass

    elif Cfg.MODEL_TYPE == 'InceptionV4':
        # Build model
        input_size = Cfg.INCEPTION_V4_INPUT_SIZE

        inception_v4_obj = InceptionV4(input_shape=input_size, classes=classes_numbers)
        model = inception_v4_obj()

        # Compile model
        optimizer = RMSprop(learning_rate=0.01, epsilon=0.1)
        model.compile(loss=CategoricalCrossentropy(), optimizer=optimizer, metrics=['accuracy'])

    elif Cfg.MODEL_TYPE == 'Inception-ResNetV1':
        # Build model
        input_size = Cfg.INCEPTION_RESNET_V1_INPUT_SIZE

        inception_resnet_v1_obj = InceptionResNetV1(input_shape=input_size, classes=classes_numbers)
        model = inception_resnet_v1_obj()

        # Optimizer
        optimizer = RMSprop(learning_rate=0.01, epsilon=0.1)

        # Compile model
        model.compile(loss=CategoricalCrossentropy(), optimizer=optimizer, metrics=['accuracy'])

    elif Cfg.MODEL_TYPE == 'Inception-ResNetV2':
        # Build model
        input_size = Cfg.INCEPTION_RESNET_V2_INPUT_SIZE

        inception_resnet_v2_obj = InceptionResNetV2(input_shape=input_size, classes=classes_numbers)
        model = inception_resnet_v2_obj()

        # Optimizer
        optimizer = RMSprop(learning_rate=0.01, epsilon=0.1)

        # Compile model
        model.compile(loss=CategoricalCrossentropy(), optimizer=optimizer, metrics=['accuracy'])

    elif Cfg.MODEL_TYPE == 'Xception':
        # Build model
        input_size = Cfg.XCEPTION_INPUT_SIZE

        xception_obj = Xception(input_shape=input_size, classes=classes_numbers)
        model = xception_obj()

        # Compile model
        optimizer = SGD(learning_rate=0.01, momentum=0.9)
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

    elif Cfg.MODEL_TYPE == 'InceptionV4':
        input_shape = Cfg.INCEPTION_V4_INPUT_SIZE

        # Load model
        model = tf.keras.models.load_model(model_path)

    elif Cfg.MODEL_TYPE == 'Inception-ResNetV1':
        input_shape = Cfg.INCEPTION_RESNET_V1_INPUT_SIZE

        # Load model
        model = tf.keras.models.load_model(model_path)

    elif Cfg.MODEL_TYPE == 'Inception-ResNetV2':
        input_shape = Cfg.INCEPTION_RESNET_V2_INPUT_SIZE

        # Load model
        model = tf.keras.models.load_model(model_path)

    elif Cfg.MODEL_TYPE == 'Xception':
        input_shape = Cfg.XCEPTION_INPUT_SIZE

        # Load model
        model = tf.keras.models.load_model(model_path)

    else:
        raise Exception('Invalid model type!')

    return model, input_shape
