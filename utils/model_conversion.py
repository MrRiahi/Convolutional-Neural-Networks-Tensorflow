import tensorflow as tf


def convert_to_tflite(model_directory):
    """
    Convert the model in model_path to TFLite.
    :param model_directory: directory of the saved model
    """

    converter = tf.lite.TFLiteConverter.from_saved_model(model_directory)
    tflite_model = converter.convert()

    return tflite_model


def save_tflite_model(tflite_model, tflite_model_directory):
    """
    Save the converted TFLite model.
    :param tflite_model: converted TFLite model
    :param tflite_model_directory: directory to save tflite_model
    """

    with open(tflite_model_directory, 'wb') as f:
        f.write(tflite_model)


def get_output_model_directory(model_directory, output_type):
    """
    Get the output directory to save the converted model.
    :param model_directory: directory of the saved model
    :param output_type: model type conversion. 'TFLite' or 'onnx'
    """

    model_name = model_directory.split('/')[-1]
    output_model_directory = f'{"/".join(model_directory.split("/")[0:-1])}/{model_name}.{output_type}'

    return output_model_directory


def convert_model(model_directory, output_type):
    """
    Convert the model to output_type.
    :param model_directory: directory of the saved model
    :param output_type: model type conversion. 'TFLite' or 'onnx'
    """

    output_model_directory = get_output_model_directory(model_directory=model_directory,
                                                        output_type=output_type)

    if output_type == 'TFLite':
        converted_model = convert_to_tflite(model_directory=model_directory)
        save_tflite_model(tflite_model=converted_model, tflite_model_directory=output_model_directory)

    elif output_type == 'onnx':
        pass

    else:
        raise Exception(f'Invalid conversion type! Can not convert model to {output_type}.')

    return output_model_directory

