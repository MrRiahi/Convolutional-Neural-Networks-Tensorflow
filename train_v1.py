import tensorflow as tf
import numpy as np

from utils.dataset import get_train_dataset, get_test_dataset, get_image_and_label
from utils.GoogLeNet_data_generator import GoogLeNetDatasetGenerator
from utils.utils import UtilityFunction
from utils.config import Config as Cfg
from utils.model import get_model


# Build model
model, input_size = get_model(classes_numbers=Cfg.CIFAR_10_CLASS_NUMBERS)

if Cfg.MODEL_TYPE == 'GoogLeNet':
    train_images_path, train_labels = get_image_and_label('./dataset/cifar-10/images/train')
    train_dataset = GoogLeNetDatasetGenerator(X_train_path=train_images_path, y_train=train_labels,
                                              batch_size=Cfg.BATCH_SIZE)

    test_images_path, test_labels = get_image_and_label('./dataset/cifar-10/images/test')
    test_dataset = GoogLeNetDatasetGenerator(X_train_path=test_images_path, y_train=test_labels,
                                             batch_size=Cfg.BATCH_SIZE)

else:
    # Get training dataset
    data_gen_args = dict(rescale=1./255,
                         rotation_range=0.2,
                         vertical_flip=True,
                         horizontal_flip=True,
                         validation_split=0.2)

    train_dataset, validation_dataset = get_train_dataset(
        directory='./dataset/cifar-10/images/train',
        aug_dict=data_gen_args,
        classes=Cfg.CIFAR_10_CLASS_NAMES,
        image_size=input_size,
        batch_size=Cfg.BATCH_SIZE,
        class_mode='categorical',
        color_mode='rgb',
        shuffle=True,
        seed=0)

    # Get test dataset
    test_dataset = get_test_dataset(
        directory='./dataset/cifar-10/images/test',
        classes=Cfg.CIFAR_10_CLASS_NAMES,
        image_size=input_size,
        batch_size=Cfg.BATCH_SIZE,
        class_mode='categorical',
        color_mode='rgb')


# def three_way(gen):
#     for x, y in gen:
#         yield x, [y, y, y]
#
#
# data_gen_args = dict(rescale=1./255,
#                      rotation_range=0.2,
#                      vertical_flip=True,
#                      horizontal_flip=True,
#                      validation_split=0.2)
#
# train_dataset, validation_dataset = get_train_dataset(
#     directory='./dataset/cifar-10/images/train',
#     aug_dict=data_gen_args,
#     classes=Cfg.CIFAR_10_CLASS_NAMES,
#     image_size=input_size,
#     batch_size=Cfg.BATCH_SIZE,
#     class_mode='categorical',
#     color_mode='rgb',
#     shuffle=True,
#     seed=0)
#
# train_dataset, validation_dataset = three_way(train_dataset), three_way(validation_dataset)

# Use callbacks
model_path = f'./models/cifar-10/{Cfg.MODEL_TYPE}'

# Use ModelCheckpoint to control validation loss for saving model.
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=model_path,
                                                               monitor='val_loss',
                                                               verbose=1,
                                                               save_best_only=True)

# Use LearningRateScheduler to decrease the learning rate during training.
learning_rate = tf.keras.callbacks.LearningRateScheduler(UtilityFunction.step_decay_classification)

callbacks = [model_checkpoint_callback, learning_rate]

# history = model.fit(train_dataset, validation_data=validation_dataset, epochs=Cfg.EPOCHS)

# Train network
if Cfg.MODEL_TYPE == 'GoogLeNet':
    history = model.fit(train_dataset, validation_data=test_dataset, epochs=Cfg.EPOCHS)

else:
    history = model.fit(train_dataset, validation_data=validation_dataset,
                        epochs=Cfg.EPOCHS, callbacks=callbacks)

# Save history
np.save(f'{model_path}/history.npy', history.history)
