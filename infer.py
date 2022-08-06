from src.config import Config as Cfg
from src.model import load_model
from src.utils import UtilityFunction as Uf


# Define model path
# model_path = f'./models/cifar-10/{Cfg.MODEL_TYPE}'
model_path = f'{Cfg.MODEL_PATH}/best'

# Load model
model, input_shape = load_model(model_path=model_path)

# Load images
images_path = ['./samples/11.png']
images = Uf.load_images(images_path=images_path, input_shape=input_shape)

# Predict
predictions = model.predict(images)

# Get label
results = Uf.get_predictions_label(predictions=predictions)

print(f'\nPrediction labels are: {results}')

