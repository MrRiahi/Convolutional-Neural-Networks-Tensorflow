import tensorflow as tf


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


def get_train_dataset(directory, classes, image_size, batch_size=128,
                      class_mode='binary', color_mode='grayscale', shuffle=True, seed=0):
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

    # train_dataset = train_datagen.map()

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


def get_test_dataset(directory, classes, image_size, batch_size=128,
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
