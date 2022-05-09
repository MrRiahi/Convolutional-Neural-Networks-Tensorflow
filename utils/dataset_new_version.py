import tensorflow as tf


def augment_data(x):
    """
    Augment the training dataset.
    :param x: input images
    """
    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip(mode="horizontal_and_vertical"),

        tf.keras.layers.RandomRotation(factor=0.2, fill_mode='constant',
                                       interpolation='bilinear', fill_value=0),

        tf.keras.layers.RandomTranslation(height_factor=0.05, width_factor=0.05, fill_mode='constant',
                                          interpolation='bilinear', fill_value=0),

    ])

    return data_augmentation(x)


def rescale_layer(x):
    """
    Rescale the input images
    :param x: input images
    """

    return tf.keras.layers.Rescaling(scale=1. / 255)(x)


def get_train_dataset(directory, classes, image_size, batch_size=128, class_mode='binary',
                      color_mode='grayscale', shuffle=True, validation_split=0.2, seed=0):
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
    :param validation_split:
    :return:
    """

    # Prepare training dataset
    train_dataset = tf.keras.preprocessing.image_dataset_from_directory(
        directory=directory,
        labels='inferred',
        label_mode=class_mode,  # 'categorical', 'binary', 'sparse', 'input'
        class_names=classes,
        color_mode=color_mode,  # "grayscale", "rgb", "rgba".
        batch_size=batch_size,
        image_size=image_size,
        shuffle=shuffle,
        seed=seed,
        validation_split=validation_split,
        subset='training')

    train_dataset = train_dataset.map(lambda x, y: (rescale_layer(x), y))

    # Augment training dataset
    train_dataset = train_dataset.map(lambda x, y: (augment_data(x), y))

    # Prepare validation dataset
    validation_dataset = tf.keras.preprocessing.image_dataset_from_directory(
        directory=directory,
        labels='inferred',
        label_mode=class_mode,  # 'categorical', 'binary', 'sparse', 'input'
        class_names=classes,
        color_mode=color_mode,  # "grayscale", "rgb", "rgba".
        batch_size=batch_size,
        image_size=image_size,
        shuffle=shuffle,
        seed=seed,
        validation_split=validation_split,
        subset='validation')

    validation_dataset = validation_dataset.map(lambda x, y: (rescale_layer(x), y))

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

    # Prepare training dataset
    test_dataset = tf.keras.preprocessing.image_dataset_from_directory(
        directory=directory,
        labels='inferred',
        label_mode=class_mode,  # 'categorical', 'binary', 'sparse', 'input'
        class_names=classes,
        color_mode=color_mode,  # "grayscale", "rgb", "rgba".
        batch_size=batch_size,
        image_size=image_size)

    test_dataset = test_dataset.map(lambda x: rescale_layer(x))

    return test_dataset
