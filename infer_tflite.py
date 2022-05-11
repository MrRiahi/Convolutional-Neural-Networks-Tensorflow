import tensorflow as tf

from utils.utils import UtilityFunction as Uf
from utils.config import Config as Cfg


# Define model path
model_path = f'./models/cifar-10/{Cfg.MODEL_TYPE}.tflite'

# Load TFLite model and allocate tensors
interpreter = tf.lite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()

# Load images
images_path = ['./samples/11.png']
images = Uf.load_images(images_path=images_path)

# Convert float64 to float32
images = images.astype('float32')

# Get input and output tensors
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Infer on a sample image
interpreter.set_tensor(input_details[0]['index'], images)
interpreter.invoke()

output_data = interpreter.get_tensor(output_details[0]['index'])
print(output_data)
