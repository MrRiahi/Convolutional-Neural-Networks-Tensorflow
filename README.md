# Convolutional-Neural-Networks
In this repository, I build different image classification CNN models from scratch and train on different datasets. 
Results and codes will be updated in the future. These models are ResNet50, MobileNetV1, and MobileNetV2. In the future, this 
repository will be updated with other convolutional neural networks.

## Install

### Clone Repository

Clone repo and install requirements.txt in a Python==3.8.3 environment, including Tensorflow==2.7.0.

```bash
git clone git@github.com:MrRiahi/Convolutional-Neural-Networks.git
cd Convolutional-Neural-Networks
```

### Virtual Environment
Python virtual environment will keep dependant Python packages from interfering with other Python projects on your
system.

```bash
python -m venv venv
source venv/bin/activate
``` 

### Requirements

Install python requirements.

```bash
(venv) pip install --upgrade pip
(venv) pip install -r requirements.txt
```

## Train 

Set your model name, number of epochs, dataset details in `utils/config.py` and run the following command:

```bash
(venv) python train.py
```

## Evaluation
To evaluate your model, set your dataset path in `evaluate.py` and run the following command in terminal:

```bash
(venv) python evaluate.py
```

## Inference
To infer your model, set your image directory in `predict.py` and run the following command in terminal:

```bash
(venv) python predict.py
```

## Result
The result of models on test dataset are reported in the following table.

|             | loss_test | acc_test |
|-------------|:---------:|:--------:|
| ResNet50    |  0.6056   |  81.71   | 
| MobileNetV1 |  0.5307   |  85.37   |
| MobileNetV2 |     ?     |    ?     |

## Convert to TFLite
You can convert the tensorflow model to TFLite by using the following command:

```bash
(venv) python convert.py
```

Afterward, you can infer the TFLite model using the following command:

```bash
(venv) python infer_tflite.py
```

## Convert to Onnx
You can convert the tensorflow model to Onnx by using the following command in terminal:

```bash
(venv) python -m tf2onnx.convert --saved-model models/cifar-10/ResNet50 \
              --output models/cifar-10/ResNet50.onnx --opset 11 --verbos
```

# TODO
- [x] Train ResNet50
- [x] Train MobileNetV1
- [ ] Train MobileNetV2
- [ ] Train GoogleNet
- [x] Add evaluation 
- [x] Add Inference
- [x] Convert models to TFLite
- [x] Convert  models to Onnx
- [x] Inference with TFLite
- [ ] Inference with Onnx

# References
* ResNet50: https://arxiv.org/pdf/1512.03385v1.pdf
* MobileNetV1: https://arxiv.org/pdf/1704.04861.pdf
* MobileNetV2: https://arxiv.org/pdf/1801.04381.pdf




