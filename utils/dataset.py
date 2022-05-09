import tensorflow as tf


def get_train_dataset(directory, aug_dict, classes, image_size, batch_size=128,
                      class_mode='binary', color_mode='grayscale', shuffle=True, seed=0):
    """
    This function creates an object from image_dataset_from_directory for training
    :param directory:
    :param aug_dict:
    :param classes:
    :param image_size:
    :param batch_size:
    :param class_mode:
    :param color_mode:
    :param shuffle:
    :param seed:
    :return:
    """

    train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(**aug_dict)

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
