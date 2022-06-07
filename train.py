import tensorflow as tf
import numpy as np

from utils.dataset import get_dataset
from utils.utils import UtilityFunction
from utils.config import Config as Cfg
from utils.model import get_model


# Build and compile model
model, input_size = get_model(classes_numbers=Cfg.CIFAR_10_CLASS_NUMBERS)

# Get train and val datasets
train_dataset, val_dataset = get_dataset(input_size=input_size)

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
history = model.fit(train_dataset, validation_data=val_dataset,
                    epochs=Cfg.EPOCHS, callbacks=callbacks)

# Save history
np.save(f'{model_path}/history.npy', history.history)
