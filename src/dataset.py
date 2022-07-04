from src.config import Config as Cfg

import tensorflow as tf
import numpy as np
import os


def get_image_data_generator(rescale=1./255, rotation_range=40, vertical_flip=True, horizontal_flip=True,
                             height_shift_range=0.2, width_shift_range=0.2, fill_mode='nearest',
                             cval=0, validation_split=0.2):
    """
    This function creates an image generator.
    :param rescale: rescaling factor
    :param rotation_range: Degree range for random rotations in int.
    :param vertical_flip: Boolean. Randomly flip inputs vertically.
    :param horizontal_flip: Boolean. Randomly flip inputs horizontally.
    :param height_shift_range: Float, 1-D array-like or int
    :param width_shift_range: Float, 1-D array-like or int
    :param fill_mode: One of {"constant", "nearest", "reflect" or "wrap"}. Default is 'nearest'.
                      Points outside the boundaries of the input are filled according to the given mode
    :param cval: Float or Int. Value used for points outside the boundaries when fill_mode = "constant"
    :param validation_split: Float. Fraction of images reserved for validation (strictly between 0 and 1).
    """

    image_data_generator = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=rescale, rotation_range=rotation_range, vertical_flip=vertical_flip,
        horizontal_flip=horizontal_flip, height_shift_range=height_shift_range,
        width_shift_range=width_shift_range, fill_mode=fill_mode, cval=cval,
        validation_split=validation_split)

    return image_data_generator


def get_train_dataset_with_image_data_gen(directory, classes, image_size, batch_size=128,
                                          class_mode='binary', color_mode='grayscale', shuffle=True,
                                          seed=0):
    """
    This function creates an object from image_dataset_from_directory for training
    :param directory:
    :param classes:
    :param image_size:
    :param batch_size:
    :param class_mode:
    :param color_mode:
    :param shuffle:
    :param seed:
    :return:
    """

    train_datagen = get_image_data_generator()

    train_dataset = train_datagen.flow_from_directory(
        directory=directory,
        target_size=image_size,
        color_mode=color_mode,  # 'grayscale', 'rgb', 'rgba'
        classes=classes,
        class_mode=class_mode,  # 'categorical', 'binary', 'sparse', 'input'
        batch_size=batch_size,
        shuffle=shuffle,
        subset='training',
        seed=seed)

    validation_dataset = train_datagen.flow_from_directory(
        directory=directory,
        target_size=image_size,
        color_mode=color_mode,  # 'grayscale', 'rgb', 'rgba'
        classes=classes,
        class_mode=class_mode,  # 'categorical', 'binary', 'sparse', 'input'
        batch_size=batch_size,
        shuffle=shuffle,
        subset='validation',
        seed=seed)

    return train_dataset, validation_dataset


def get_test_dataset_with_image_data_gen(directory, classes, image_size, batch_size=128,
                                         class_mode='binary', color_mode='grayscale'):
    """
    This function creates an object from image_dataset_from_directory for training
    :param directory:
    :param classes:
    :param image_size:
    :param batch_size:
    :param class_mode:
    :param color_mode:
    :return:
    """

    test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

    test_dataset = test_datagen.flow_from_directory(
        directory=directory,
        target_size=image_size,
        color_mode=color_mode,  # 'grayscale', 'rgb', 'rgba'
        classes=classes,
        class_mode=class_mode,  # 'categorical', 'binary', 'sparse', 'input'
        batch_size=batch_size)

    return test_dataset


# --------------------- GoogLeNet data generator --------------------- #


def process_image(image_path):
    """
    Load an image.
    :param image_path: path of image
    :return:
    """

    image = tf.io.read_file(filename=image_path)
    image = tf.image.decode_png(contents=image, channels=3)
    image = tf.image.convert_image_dtype(image=image, dtype=tf.float32)

    image = tf.image.resize(images=image, size=Cfg.MOBILENET_V2_INPUT_SIZE, method='nearest')

    return image


def process_label(on_hot_label):
    """
    Repeat the one_hot_label 3 time for GoogLeNet network
    :param on_hot_label: one hot encoding label
    :return:
    """

    one_hot_3d_label = [on_hot_label, on_hot_label, on_hot_label]

    return one_hot_3d_label


def get_images_list(directory):
    """
    Get images in the image_directory path with their labels and put them in lists.
    """

    images_list = []
    classes_list = []

    for folder in os.listdir(directory):
        images_path = f'{directory}/{folder}'

        images = [f'{images_path}/{image_path}' for image_path in os.listdir(images_path)]
        images_list.extend(images)

        classes = [tf.one_hot(Cfg.CIFAR10_CLASS_NAME_TO_NUMBER[folder], Cfg.CIFAR_10_CLASS_NUMBERS)
                   for _ in range(len(images))]
        classes_list.extend(classes)

    return np.array(images_list), np.array(classes_list)


def prepare_dataset(image_list, label_list, is_train=True):
    """
    Prepare dataset from image_list and label_list
    :param image_list: list of images
    :param label_list: list of one hot labels
    :param is_train: a flag to specify whether using data augmentation (for train subset) or not (for val subset)
    :return:
    """
    image_filenames = tf.constant(image_list)

    # Train subset
    slices_dataset = tf.data.Dataset.from_tensor_slices(image_filenames)
    slices_labels = tf.data.Dataset.from_tensor_slices(label_list)

    if is_train:
        image_dataset = slices_dataset.map(map_func=process_image).\
            map(lambda image: tf.image.stateless_random_flip_left_right(image=image, seed=(2, 5))).\
            map(lambda image: tf.image.stateless_random_flip_up_down(image=image, seed=(10, 11)))
    else:
        image_dataset = slices_dataset.map(map_func=process_image)

    label_dataset = slices_labels.map(map_func=process_label)

    x_dataset = image_dataset.shuffle(buffer_size=Cfg.BUFFER_SIZE, seed=0).\
        batch(batch_size=Cfg.BATCH_SIZE)
    y_dataset = label_dataset.shuffle(buffer_size=Cfg.BUFFER_SIZE, seed=0).\
        batch(batch_size=Cfg.BATCH_SIZE)

    dataset = tf.data.Dataset.zip((x_dataset, y_dataset))

    return dataset


def get_train_dataset_with_tf_dataset():
    """
    This function reads the images and masks names in a list and load.
    :return:
    """

    train_directory = f'./{Cfg.TRAIN_DATASET_PATH}'

    images_list, classes_list = get_images_list(directory=train_directory)

    shuffled_indices = np.random.permutation(len(images_list))

    shuffled_image_list = images_list[shuffled_indices]
    shuffled_classes_list = classes_list[shuffled_indices]

    train_part = int(np.floor(Cfg.TRAIN_SUBSET * len(images_list)))

    # Train part
    train_list = shuffled_image_list[:train_part]
    image_train_filenames = tf.constant(train_list)

    classes_train_list = shuffled_classes_list[:train_part]

    train_dataset = prepare_dataset(image_list=image_train_filenames, label_list=classes_train_list,
                                    is_train=True)

    # Validation
    val_list = shuffled_image_list[train_part:]
    image_val_filenames = tf.constant(val_list)

    classes_val_list = shuffled_classes_list[train_part:]

    val_dataset = prepare_dataset(image_list=image_val_filenames, label_list=classes_val_list,
                                  is_train=False)

    return train_dataset, val_dataset


def get_test_dataset_with_tf_dataset():
    """
    This function reads the images and masks names in a list and load for test.
    :return:
    """

    train_directory = f'./{Cfg.TEST_DATASET_PATH}'

    images_list, classes_list = get_images_list(directory=train_directory)

    # Test part
    image_filenames = tf.constant(images_list)

    test_dataset = prepare_dataset(image_list=image_filenames, label_list=classes_list,
                                   is_train=False)

    return test_dataset


def get_train_dataset(input_shape):
    """
    Load train dataset.
    :param input_shape: model input shape
    :return:
    """

    if Cfg.MODEL_TYPE == 'GoogLeNet':
        train_dataset, val_dataset = get_train_dataset_with_tf_dataset()

    elif Cfg.MODEL_TYPE in ['ResNet50', 'MobileNetV1', 'MobileNetV2', 'VGG16', 'VGG13',
                            'VGG11', 'BNInception']:
        train_dataset, val_dataset = get_train_dataset_with_image_data_gen(
            directory=f'./{Cfg.TRAIN_DATASET_PATH}',
            classes=Cfg.CIFAR_10_CLASS_NAMES,
            image_size=input_shape,
            batch_size=Cfg.BATCH_SIZE,
            class_mode='categorical',
            color_mode='rgb',
            shuffle=True,
            seed=0)

    else:
        raise Exception('Invalid model type!')

    return train_dataset, val_dataset


def get_test_dataset(input_shape):
    """
    Load test dataset.
    :param input_shape: model input shape
    :return:
    """

    if Cfg.MODEL_TYPE == 'GoogLeNet':
        test_dataset = get_test_dataset_with_tf_dataset()

    elif Cfg.MODEL_TYPE in ['ResNet50', 'MobileNetV1', 'MobileNetV2', 'VGG16', 'VGG13',
                            'VGG11', 'BNInception']:
        test_dataset = get_test_dataset_with_image_data_gen(
            directory=f'./{Cfg.TEST_DATASET_PATH}',
            classes=Cfg.CIFAR_10_CLASS_NAMES,
            image_size=input_shape,
            batch_size=Cfg.BATCH_SIZE,
            class_mode='categorical',
            color_mode='rgb')

    else:
        raise Exception('Invalid model type!')

    return test_dataset
