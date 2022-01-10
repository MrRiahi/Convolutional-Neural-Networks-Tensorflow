from tensorflow.keras.losses import CategoricalCrossentropy
import tensorflow_datasets as tfd

from config import Config as Cfg
from ResNet import ResNet50
from MobileNet import MobileNetV1
from preprocess import preprocess_caltech101

# Load caltech101 dataset
(ds_train, ds_test), ds_info = tfd.load('caltech101', split=['train', 'test'], shuffle_files=True,
                                        as_supervised=True, with_info=True)

train_ds = (ds_train.shuffle(Cfg.BATCH_SIZE * 100).map(preprocess_caltech101, num_parallel_calls=Cfg.AUTO)
            .batch(Cfg.BATCH_SIZE).prefetch(Cfg.AUTO))
ds_test = (ds_test.map(preprocess_caltech101, num_parallel_calls=Cfg.AUTO).batch(Cfg.BATCH_SIZE)
           .prefetch(Cfg.AUTO))

# Load model and compile it
# res_net = ResNet50(input_shape=Cfg.RESNET50_INPUT_SIZE, classes=Cfg.CLASSES)
# model = res_net.res_net50()

mobile_net = MobileNetV1(input_shape=Cfg.MOBILENET_V1_INPUT_SIZE, classes=Cfg.CLASSES)
model = mobile_net.mobile_net_v1()

model.compile(loss=CategoricalCrossentropy, optimizer='adam', metrics=["accuracy"])

# Train network
model.fit(train_ds, validation_data=ds_test, epochs=Cfg.EPOCHS)
