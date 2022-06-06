import tflite_runtime.interpreter as tflite
import numpy as np

from utils.utils import UtilityFunction as Uf
from utils.config import Config as Cfg


# Define model path
model_path = f'./models/cifar-10/{Cfg.MODEL_TYPE}.tflite'

# Load TFLite model and allocate tensors
interpreter = tflite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()

# Get input and output tensors
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Get input shape
input_shape = tuple(input_details[0]['shape'][1:3])

# Load images
images_path = ['./samples/30.png']
images = Uf.load_images(images_path=images_path, input_shape=input_shape)

# Convert float64 to float32
images = images.astype('float32')

# Infer on a sample image
interpreter.set_tensor(input_details[0]['index'], images)
interpreter.invoke()

output_data = interpreter.get_tensor(output_details[0]['index'])

# Convert output to corresponding label
label = Cfg.CIFAR_10_CLASS_NAMES[np.argmax(output_data[0])]

print(f'The predicted labels is {label}')
