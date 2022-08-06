import tensorflow as tf

from src.dataset import get_train_dataset
from src.utils import UtilityFunction
from src.config import Config as Cfg
from src.model import load_model


# Build and compile model
model, input_shape = load_model(model_path=f'{Cfg.MODEL_PATH}/last')

# Get train and val datasets
train_dataset, val_dataset = get_train_dataset(input_shape=input_shape)

# Use ModelCheckpoint to control validation loss for saving the best model.
best_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=Cfg.MODEL_PATH + '/best.val_loss{val_loss:.2f}',
                                                              monitor='val_loss',
                                                              verbose=1,
                                                              save_best_only=True)

# Use ModelCheckpoint to save last checkpoint
last_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=f'{Cfg.MODEL_PATH}/last',
                                                              verbose=1,
                                                              save_freq='epoch')
# Use LearningRateScheduler to decrease the learning rate during training.
learning_rate = tf.keras.callbacks.LearningRateScheduler(UtilityFunction.learning_rate_decay)

callbacks = [best_checkpoint_callback, last_checkpoint_callback, learning_rate]

# Train network
history = model.fit(train_dataset, validation_data=val_dataset,
                    epochs=Cfg.EPOCHS, callbacks=callbacks)

# Save history
UtilityFunction.save_history(history=history.history)
