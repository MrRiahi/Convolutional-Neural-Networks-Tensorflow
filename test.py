from utils.dataset import get_test_dataset
from utils.config import Config as Cfg
from utils.model import load_model


# Define model path
model_path = f'./models/cifar-10/{Cfg.MODEL_TYPE}'

# Load model
model, input_shape = load_model(model_path=model_path)

# Get test dataset
test_dataset = get_test_dataset(
    directory='./dataset/cifar-10/images/test',
    classes=Cfg.CIFAR_10_CLASS_NAMES,
    image_size=input_shape,
    batch_size=Cfg.BATCH_SIZE,
    class_mode='categorical',
    color_mode='rgb')

# Evaluate model on test data
eval_results = model.evaluate(test_dataset)

print(eval_results)
