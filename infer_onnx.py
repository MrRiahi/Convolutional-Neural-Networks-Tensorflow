import numpy as np
import onnxruntime as ort

from utils.utils import UtilityFunction as Uf
from utils.config import Config as Cfg


# Define model path
model_path = f'./models/cifar-10/{Cfg.MODEL_TYPE}.onnx'

# Load Onnx model
ort_sess = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])

# Get input shape
input_shape = tuple(ort_sess.get_inputs()[0].shape[1:3])

# Load images
images_path = ['./samples/30.png']
images = Uf.load_images(images_path=images_path, input_shape=input_shape)

# Convert float64 to float32
images = images.astype('float32')

# Infer on a sample image
input_name = ort_sess.get_inputs()[0].name
output = ort_sess.run(None, {input_name: images})

# Convert output to corresponding label
label = Cfg.CIFAR_10_CLASS_NAMES[np.argmax(output[0][0])]

print(f'The predicted labels is {label}')
