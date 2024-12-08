import itertools

import matplotlib.pyplot as plt
import pickle
import numpy as np
from sklearn.metrics import confusion_matrix

savedir = 'result/'
plt.rcParams["font.sans-serif"]="SimHei"
plt.rcParams["axes.unicode_minus"] = False


def plot_withval(history, i, savepath='./result'):
    # 绘制损失与准确率曲线
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    specificity = history.history['metric_specificity']
    val_specificity = history.history['val_metric_specificity']
    recall = history.history['recall']
    val_recall = history.history['val_recall']
    epochs = range(1, len(acc) + 1)

    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.savefig(savepath + '/{}_acc.png'.format(i))

    plt.figure()
    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    plt.savefig(savepath + '/{}_loss.png'.format(i))

    plt.figure()

    plt.plot(epochs, specificity, 'bo', label='Training specificity')
    plt.plot(epochs, val_specificity, 'b', label='Validation specificity')
    plt.title('Training and validation specificity')
    plt.legend()
    plt.savefig(savepath + '/{}_specificity.png'.format(i))
    plt.figure()

    plt.plot(epochs, recall, 'bo', label='Training recall')
    plt.plot(epochs, val_recall, 'b', label='Validation recall')
    plt.title('Training and validation recall')
    plt.legend()
    plt.savefig(savepath + '/{}_recall.png'.format(i))

    plt.show()


def plot_metric(x, metric,metric_name, i,savepath):
    plt.figure()
    plt.plot(x, metric, 'b', label='Training {}'.format(metric_name))
    plt.title('Training  {}'.format(metric_name))
    plt.legend()
    plt.savefig(savepath+ '/{}f_{}.png'.format(i, metric_name))


def plot(history,i,savepath):
    acc = history.history['acc']
    loss = history.history['loss']
    specificity = history.history['metric_specificity']
    recall = history.history['recall']

    epochs = range(1, len(acc) + 1)

    plot_metric(epochs, acc, 'acc',i,savepath)
    plot_metric(epochs, loss, 'loss',i,savepath)
    plot_metric(epochs, specificity, 'specificity',i,savepath)
    plot_metric(epochs, recall, 'recall',i,savepath)

    fig = plt.figure()
    ax1 = fig.add_subplot(2,2,1)
    ax1.plot(epochs, acc, 'b', label='Training acc')
    ax1.legend()
    ax2 = fig.add_subplot(2,2,2)
    ax2.plot(epochs, loss, 'b', label='Training loss')
    ax2.legend()
    ax3 = fig.add_subplot(2, 2, 3)
    ax3.plot(epochs, specificity, 'b', label='Training specificity')
    ax3.legend()
    ax4 = fig.add_subplot(2,2,4)
    ax4.plot(epochs, recall, 'b', label='Training recall')
    ax4.legend()
    fig.savefig(savepath + '/{}f_all.png'.format(i))
    plt.show()


def plot_one(history, i, savepath,label):
    acc = history.history['acc']
    loss = history.history['loss']
    specificity = history.history['metric_specificity']
    recall = history.history['recall']

    epochs = range(1, len(acc) + 1)

    plt.plot(epochs, acc, '*-', label='Training acc')
    plt.plot(epochs, loss, 'rx-', label='Training loss')
    plt.plot(epochs, specificity, 'go-', label='Training specificity')
    plt.plot(epochs, recall, 'b-.', label='Training recall')
    plt.title('Fold{} multiindex'.format(i))
    plt.xlabel('Epochs')

    plt.legend()
    plt.savefig(savepath + 'result/{}f_together_{}.png'.format(i,label))
    # plt.show()
    plt.close()


def plot_roc_curve(history, i):
    acc = history.history['acc']
    loss = history.history['loss']
    specificity = history.history['metric_specificity']
    recall = history.history['recall']
    fpr = history.history['metric_FPR']
    plt.plot(fpr, recall,'b')
    plt.title('Fold{} ROC Curve'.format(i))
    plt.xlabel('1 - Specificity')
    plt.ylabel('Sensitivity')
    plt.xticks(np.arange(0, 1, 0.25))
    plt.yticks(np.arange(0, 1, 0.25))
    # plt.legend()
    plt.savefig('result/{}f_roc'.format(i))
    plt.show()


def plot_history():
    with open('history/i3d_model_3f_20history.pickle', 'rb') as file:
        history = pickle.load(file)
    acc = history['acc']
    loss = history['loss']
    specificity = history['metric_specificity']
    recall = history['recall']
    epochs = range(1, len(acc) + 1)

    plt.plot(epochs, acc, 'b', label='Training acc')
    plt.title('Training  accuracy')
    plt.legend()
    plt.show()

def plot_confusion_matrix(cm, classes,
                          title='Confusion matrix',
                          cmap=plt.cm.jet):
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, '{:.2f}'.format(cm[i, j]), horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

# 显示混淆矩阵
def plot_confuse(model, x_val, y_val):
    predictions = model.predict_classes(x_val)
    truelabel = y_val.argmax(axis=-1)   # 将one-hot转化为label
    conf_mat = confusion_matrix(y_true=truelabel, y_pred=predictions)
    plt.figure()
    plot_confusion_matrix(conf_mat, range(np.max(truelabel)+1))
