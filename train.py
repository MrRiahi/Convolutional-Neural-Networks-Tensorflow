from tensorflow.keras.losses import CategoricalCrossentropy
import tensorflow as tf
import numpy as np

from utils.dataset import get_train_dataset, get_test_dataset
from utils.utils import UtilityFunction
from utils.config import Config as Cfg
from utils.model import get_model


# Build model
model, input_size = get_model()

# Compile model
model.compile(loss=CategoricalCrossentropy(), optimizer='adam', metrics=["accuracy"])

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

# Train network
history = model.fit(train_dataset, validation_data=validation_dataset,
                    epochs=Cfg.EPOCHS, callbacks=callbacks)

# Save history
np.save(f'{model_path}/history.npy', history.history)
