from utils.model_conversion import convert_model
from utils.config import Config as Cfg


# Define model path
model_path = f'./models/cifar-10/{Cfg.MODEL_TYPE}'

# Convert and save model to 'TFLite' or 'onnx'
output_model_directory = convert_model(model_directory=model_path, output_type='TFLite')

print(f'Model saved in {output_model_directory} directory')
