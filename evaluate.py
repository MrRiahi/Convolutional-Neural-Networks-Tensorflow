from utils.dataset import get_test_dataset
from utils.config import Config as Cfg
from utils.model import load_model


# Define model path
model_path = f'./models/cifar-10/{Cfg.MODEL_TYPE}'

# Load model
model, input_shape = load_model(model_path=model_path)
print(f'Model loads from {model_path}')

# Get test dataset
test_dataset = get_test_dataset(input_shape=input_shape)

# Evaluate model on test data
eval_results = model.evaluate(test_dataset)

if Cfg.MODEL_TYPE == 'GoogLeNet':
    print(f'{Cfg.MODEL_TYPE} loss: {round(eval_results[1], ndigits=4)}'
          f'\n{Cfg.MODEL_TYPE} accuracy: {round(100 * eval_results[4], ndigits=2)}')
else:
    print(f'{Cfg.MODEL_TYPE} loss: {round(eval_results[0], ndigits=4)}'
          f'\n{Cfg.MODEL_TYPE} accuracy: {round(100 * eval_results[1], ndigits=2)}')

