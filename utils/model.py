from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.optimizers import SGD

from utils.MobileNet import MobileNetV1, MobileNetV2
from utils.config import Config as Cfg
from utils.ResNet import ResNet50
from utils.GoogLeNet import GoogLeNet


def get_model(classes_numbers):
    """
    This function builds the model defined in config.py file
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
        model.compile(loss=CategoricalCrossentropy(), optimizer='adam', metrics=['accuracy'])

    elif Cfg.MODEL_TYPE == 'GoogLeNet':
        # Build model
        input_size = Cfg.GOOGLE_NET_INPUT_SIZE
        google_net = GoogLeNet(input_shape=input_size, classes=classes_numbers)
        model = google_net.google_net()

        # Compile model
        losses = {'output': CategoricalCrossentropy(), 'output_aux_1': CategoricalCrossentropy(),
                  'output_aux_2': CategoricalCrossentropy()}

        metrics = {'output': 'accuracy', 'output_aux_1': 'accuracy', 'output_aux_2': 'accuracy'}

        sgd = SGD(learning_rate=0.1, momentum=0.9)

        model.compile(loss=losses, loss_weights=[1, 0.3, 0.3], optimizer=sgd, metrics=metrics)

    else:
        raise Exception('Invalid model type')

    return model, input_size
