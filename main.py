from tensorflow.keras.losses import CategoricalCrossentropy
import tensorflow_datasets as tfd

from config import Config
from ResNet import ResNet50
from preprocess import preprocess_caltech101

# Load caltech101 dataset
(ds_train, ds_test), ds_info = tfd.load('caltech101', split=['train', 'test'], shuffle_files=True,
                                        as_supervised=True, with_info=True)

train_ds = (ds_train.shuffle(Config.BATCH_SIZE * 100).map(preprocess_caltech101, num_parallel_calls=Config.AUTO)
            .batch(Config.BATCH_SIZE).prefetch(Config.AUTO))
ds_test = (ds_test.map(preprocess_caltech101, num_parallel_calls=Config.AUTO).batch(Config.BATCH_SIZE)
           .prefetch(Config.AUTO))

# Load model and compile it
res_net = ResNet50(input_shape=Config.INPUT_SIZE, classes=Config.CLASSES)
model = res_net.res_net50()

model.compile(loss=CategoricalCrossentropy(label_smoothing=0.1), optimizer='adam', metrics=["accuracy"])

# Train network
model.fit(train_ds, validation_data=ds_test, epochs=Config.EPOCHS)
