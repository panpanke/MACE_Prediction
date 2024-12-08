import matplotlib.pyplot as plt

savedir = '/result/'
plt.rcParams["axes.unicode_minus"] = False


def plot_one(history, i, label, save_path):
    acc = history.history['acc']
    loss = history.history['loss']
    specificity = history.history['metric_specificity']
    recall = history.history['recall']

    epochs = range(1, len(acc) + 1)

    plt.plot(epochs, acc, '*-', label='Training acc')
    plt.plot(epochs, loss, 'rx-', label='Training loss')
    plt.plot(epochs, specificity, 'go-', label='Training specificity')
    plt.plot(epochs, recall, 'b-.', label='Training recall')
    plt.title('Multi-index change chart of Fold{}'.format(i))
    plt.xlabel('Epochs')

    plt.legend()
    plt.savefig(save_path + '/{}f_together_{}.png'.format(i, label))
    # plt.show()
    plt.close()
