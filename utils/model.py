from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.optimizers import SGD
import tensorflow as tf

from utils.MobileNet import MobileNetV1, MobileNetV2
from utils.config import Config as Cfg
from utils.ResNet import ResNet50
from utils.GoogLeNet import GoogLeNet


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
        mobile_net_v2 = MobileNetV2(input_shape=input_size, class_numbers=classes_numbers)
        model = mobile_net_v2.mobile_net_v2()

        # Compile model
        model.compile(loss=CategoricalCrossentropy(), optimizer='sgd', metrics=['accuracy'])

    elif Cfg.MODEL_TYPE == 'GoogLeNet':
        # Build model
        input_size = Cfg.GOOGLE_NET_INPUT_SIZE
        google_net = GoogLeNet(input_shape=input_size, classes=classes_numbers)
        model = google_net.google_net()

        optimizer = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9,
                                            nesterov=False)

        # Compile model
        losses = {'output': CategoricalCrossentropy(), 'output_aux_1': CategoricalCrossentropy(),
                  'output_aux_2': CategoricalCrossentropy()}

        metrics = {'output': 'accuracy', 'output_aux_1': 'accuracy', 'output_aux_2': 'accuracy'}

        optimizer = SGD(learning_rate=0.1, momentum=0.9)

        model.compile(loss=losses,
                      optimizer=optimizer,
                      loss_weights=[1, 0.3, 0.3],
                      metrics=metrics)

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

    else:
        raise Exception('Invalid model type!')

    return model, input_shape
