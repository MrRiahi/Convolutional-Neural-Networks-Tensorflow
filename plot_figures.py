import matplotlib.pyplot as plt
import numpy as np
import os

models_path = f'./models/cifar-10/'
models_name = [model_name for model_name in os.listdir(models_path) if len(model_name.split('.')) == 1]

fig_train_acc, ax_train_acc = plt.subplots()
ax_train_acc.set_ylabel('Accuracy')
ax_train_acc.set_xlabel('Epochs')
ax_train_acc.set_title('Train accuracy')

fig_train_loss, ax_train_loss = plt.subplots()
ax_train_loss.set_ylabel('Loss')
ax_train_loss.set_xlabel('Epochs')
ax_train_loss.set_title('Train loss')

fig_val_acc, ax_val_acc = plt.subplots()
ax_val_acc.set_ylabel('Accuracy')
ax_val_acc.set_xlabel('Epochs')
ax_val_acc.set_title('Validation accuracy')

fig_val_loss, ax_val_loss = plt.subplots()
ax_val_loss.set_ylabel('Loss')
ax_val_loss.set_xlabel('Epochs')
ax_val_loss.set_title('Validation loss')

for model_name in models_name:
    history = np.load(f'{models_path}/{model_name}/history.npy', allow_pickle=True)
    history = history.tolist()

    train_loss = history['loss']
    val_loss = history['val_loss']

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



