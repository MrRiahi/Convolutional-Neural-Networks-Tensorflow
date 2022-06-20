import os
import cv2
import pickle
import tensorflow as tf

from src.config import Config as Cfg


def preprocess_caltech101(image, label):
    image = tf.image.resize(image, (Cfg.INPUT_SIZE[0], Cfg.INPUT_SIZE[1]))
    label = tf.one_hot(label, depth=Cfg.CLASSES)

    return image, label


def preprocess_cifar10(image, label):
    image = tf.image.resize(image, (Cfg.MOBILENET_V1_INPUT_SIZE[0], Cfg.MOBILENET_V1_INPUT_SIZE[1]))
    label = tf.one_hot(label, depth=Cfg.CLASSES)

    return image, label


def unpickle_cifar(file):
    """
    This function loads the cifar dataset data
    :param file:
    :return:
    """
    with open(file, 'rb') as fo:
        dataset_dict = pickle.load(fo, encoding='bytes')

    return dataset_dict


def make_cifar_directories(meta_path):
    """
    This function makes the directories for cifar dataset
    :param meta_path:
    :return:
    """

    meta = unpickle_cifar(meta_path)

    label_names = meta[b'label_names']

    class_names = {}

    for cnt, label_name in enumerate(label_names):
        label_name = label_name.decode('utf-8')
        class_names[cnt] = label_name

        path_train = f'../dataset/cifar-10/images/train/{label_name}'
        path_test = f'../dataset/cifar-10/images/test/{label_name}'

        os.makedirs(path_train, exist_ok=True)
        os.makedirs(path_test, exist_ok=True)

    return class_names


def get_cifar_class_name(class_id, names_dict):
    """
    This function gets the class name of label
    :param class_id:
    :param names_dict:
    :return:
    """

    return names_dict[class_id]


def save_cifar_images():
    """
    This function saves the images in data_batch_1 to data_batch_5
    :return:
    """

    files_path = ['../dataset/cifar-10/cifar-10/data_batch_1', '../dataset/cifar-10/cifar-10/data_batch_2',
                  '../dataset/cifar-10/cifar-10/data_batch_3', '../dataset/cifar-10/cifar-10/data_batch_4',
                  '../dataset/cifar-10/cifar-10/data_batch_5']

    names_dict = make_cifar_directories(meta_path='../dataset/cifar-10/cifar-10/batches.meta')
    cnt = 1
    for file_path in files_path:
        data_dict = unpickle_cifar(file=file_path)

        for label, data, name in zip(data_dict[b'labels'], data_dict[b'data'], data_dict[b'filenames']):

            image = data.reshape(3, 32, 32)
            image = image.transpose(1, 2, 0)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB

            class_name = get_cifar_class_name(class_id=label, names_dict=names_dict)

            image_path = f'../dataset/cifar-10/images/train/{class_name}/{cnt}.png'
            cv2.imwrite(image_path, image)

            cnt += 1

    # Test images
    file_path = '../dataset/cifar-10/cifar-10/test_batch'

    data_dict = unpickle_cifar(file=file_path)
    cnt = 1
    for label, data, name in zip(data_dict[b'labels'], data_dict[b'data'], data_dict[b'filenames']):

        image = data.reshape(3, 32, 32)
        image = image.transpose(1, 2, 0)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB

        class_name = get_cifar_class_name(class_id=label, names_dict=names_dict)

        image_path = f'../dataset/cifar-10/images/test/{class_name}/{cnt}.png'
        cv2.imwrite(image_path, image)

        cnt += 1


if __name__ == "__main__":
    save_cifar_images()
