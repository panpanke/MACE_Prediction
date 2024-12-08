# encoding: utf-8
import argparse

import keras
import numpy as np
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from i3d_model import Inception_Inflated3d
from keras import optimizers
from util.keras_metrics import metric_specificity, recall, fbeta_score

from plot import plot_one
from mygenerator import data_generator
import glob
from predict_cine import write_default, write_result

FRAME_HEIGHT = 128
FRAME_WIDTH = 128
NUM_RGB_CHANNELS = 3
NUM_CLASSES = 2


def train(list_0, list_1, args, label):
    n = args.i
    print("fold{} start training...".format(n))
    # load data
    dataset = []
    imglist = list_0 + list_1
    for i in range(int(len(imglist) / 30)):
        dataset.append(imglist[i * 30:i * 30 + 30])
    labels = ['0'] * int(len(list_0) / 30) + ['1'] * int(len(list_1) / 30)

    seed = 28
    np.random.seed(seed)
    np.random.shuffle(labels)
    np.random.seed(seed)
    np.random.shuffle(dataset)
    print('train numbers:', len(labels))

    width, height = 128, 128
    IMAGE_SIZE = (1, width, height, 3)
    train_gen = data_generator(dataset, labels, IMAGE_SIZE, batchsize=args.batchsize)
    keras.initializers.he_normal()
    rgb_model = Inception_Inflated3d(
        include_top=False,
        # weights='rgb_kinetics_only',
        input_shape=(30, 128, 128, 3),
        classes=NUM_CLASSES)

    optimizer = optimizers.Adam(lr=1e-4)
    rgb_model.compile(loss='binary_crossentropy',
                      optimizer=optimizer,
                      metrics=['acc',
                               recall,
                               metric_specificity,
                               fbeta_score])

    callback_list = [
        EarlyStopping(monitor='loss',
                      patience=5,
                      mode='min',
                      min_delta=1e-3),
        TensorBoard(log_dir='./tf_log',
                    batch_size=args.batchsize,
                    write_grads=False),
        ModelCheckpoint(
            filepath=save_path + '/model/cine_model.h5',
            monitor='recall',
            mode='max',
            save_best_only=True, ),
        # ReduceLROnPlateau(monitor='recall', patience=5,
        #                               verbose=1,factor=0.2, min_lr=1e-7)
    ]

    steps = len(dataset) // train_gen.batch_size
    history = rgb_model.fit_generator(generator=train_gen.get_mini_batch(),
                                      steps_per_epoch=steps,
                                      epochs=args.epochs,
                                      callbacks=callback_list,
                                      verbose=0,
                                      )
    # rgb_model.save('E:/kepanpan/11.18/cine/model/sa_model.h5')

    # 绘制图
    plot_path = save_path + '/result'
    plot_one(history, n, label, save_path=plot_path)


def train_model(args):
    # test_list,p_num,label = [],0,0
    n = args.i
    n = str(n)
    n = n.zfill(3)
    print('Num：', n)
    i = args.i

    list_1 = glob.glob(path + '/1/cine/*.jpg')
    list_0 = glob.glob(path + '/0/cine/*.jpg') + glob.glob(path + '/0/cine_1/*.jpg')

    if args.patients == 'has':
        test_list = glob.glob(path + '/1/cine/{}*.jpg'.format(n))
        print('test data;', len(test_list))

        list_1 = list(set(list_1).difference(set(test_list)))
        list_1 = sorted(list_1)

        p_num = args.has_num
        label = 1
    elif args.patients == 'none':
        test_list = glob.glob(path + '/0/cine/{}*.jpg'.format(n))
        print('test data;', len(test_list))

        list_0 = list(set(list_0).difference(set(test_list)))
        list_0 = sorted(list_0)

        p_num = args.no_num
        label = 0

    if (test_list.__len__() == 0):
        write_default(i, p_num, label, SQ_NAME)
    else:
        train(list_0=list_0, list_1=list_1, args=args, label=label)
        write_result(i, p_num, label, SQ_NAME)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="I3D train model")
    parser.add_argument('-b', '--batchsize', type=int, default='8')
    parser.add_argument('-e', '--epochs', type=int, default='20')
    parser.add_argument('-op', '--optimizers', default='Adam')
    parser.add_argument('-lr', '--learnrate', default='1e-4')
    parser.add_argument('--i', type=int, default='9')
    parser.add_argument('-p', '--patients', type=str, choices=['has', 'none'], default='has')
    parser.add_argument('--no_num', type=int, default='428')
    parser.add_argument('--has_num', type=int, default='60')
    args = parser.parse_args()

    path = '/home/data/train'
    save_path = '/home/projects/python/MACE_model/cine/'

    SQ_NAME = 'cine'
    train_model(args)
