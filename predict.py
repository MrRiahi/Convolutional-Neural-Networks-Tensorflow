from utils.config import Config as Cfg
from utils.model import load_model
from utils.utils import UtilityFunction as Uf


# Define model path
model_path = f'./models/cifar-10/{Cfg.MODEL_TYPE}'

# Load model
model, input_shape = load_model(model_path=model_path)

# Load images
images_path = ['./samples/11.png', './samples/30.png']
images = Uf.load_images(images_path=images_path)

# Predict
predictions = model.predict(images)

# Get label
results = Uf.get_predictions_label(predictions=predictions)

print(f'\nPrediction labels are: {results}')
