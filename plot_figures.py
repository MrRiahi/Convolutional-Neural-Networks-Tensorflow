import matplotlib.pyplot as plt
import numpy as np
import os

from src.utils import UtilityFunction as Uf

models_path = f'./models/cifar-10/'
models_name = [model_name for model_name in os.listdir(models_path) if len(model_name.split('.')) == 1]


fig_train_acc, ax_train_acc = Uf.create_figure(title='Train accuracy', x_label='Epochs', y_label='Accuracy')

fig_train_loss, ax_train_loss = Uf.create_figure(title='Train loss', x_label='Epochs', y_label='Loss')


fig_val_acc, ax_val_acc = Uf.create_figure(title='Validation accuracy', x_label='Epochs', y_label='Accuracy')

fig_val_loss, ax_val_loss = Uf.create_figure(title='Validation loss', x_label='Epochs', y_label='Loss')

for model_name in models_name:
    history = np.load(f'{models_path}/{model_name}/history.npy', allow_pickle=True)
    history = history.tolist()

    if model_name == 'GoogLeNet':
        train_loss = history['output_loss']
        val_loss = history['val_output_loss']
    else:
        train_loss = history['loss']
        val_loss = history['val_loss']

    if max(val_loss) >= 10:
        val_loss = np.array(val_loss)
        val_loss[val_loss >= 10] = 10

    if model_name == 'GoogLeNet':
        train_acc = history['output_accuracy']
        val_acc = history['val_output_accuracy']
    else:
        train_acc = history['accuracy']
        val_acc = history['val_accuracy']

    ax_train_acc.plot(train_acc)
    ax_train_loss.plot(train_loss)

    ax_val_acc.plot(val_acc)
    ax_val_loss.plot(val_loss)


fig_train_acc.legend(models_name)
fig_train_loss.legend(models_name)
fig_val_acc.legend(models_name)
fig_val_loss.legend(models_name)

fig_train_acc.savefig('./images/train_acc.png')
fig_train_loss.savefig('./images/train_loss.png')
fig_val_acc.savefig('./images/val_acc.png')
fig_val_loss.savefig('./images/val_loss.png')



