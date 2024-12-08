import glob

from keras.applications import DenseNet121
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping, TensorBoard
from keras import optimizers
import argparse

from sklearn.model_selection import train_test_split

from model import make_model
from plot import plot_one
from predict import write_default, write_result
from keras.preprocessing.image import ImageDataGenerator

from util.keras_metrics import metric_specificity, recall, fbeta_score
from util.load_data import select_test_num, data_path, make_data

SQ_NAME = 't2'


def train(args, train_data, train_y, x_val, y_val, SQ_NAME):
    base_model = DenseNet121(include_top=False, weights='imagenet', input_shape=(128, 128, 3))
    model = make_model(base_model)
    callback_list = [
        EarlyStopping(monitor='recall',
                      patience=5,
                      mode='max',
                      min_delta=1e-3),
        TensorBoard(log_dir='./tf_logs',
                    batch_size=args.batchsize,
                    write_grads=False),
        ModelCheckpoint(
            filepath='{}/model/{}_model.h5'.format(save_path, SQ_NAME),
            monitor='recall',
            mode='max',
            save_best_only=True, ),
        # ReduceLROnPlateau(monitor='loss', patience=5,
        #                               verbose=1,factor=0.2, min_lr=1e-7)
    ]
    print('Fold {} training...'.format(args.i))

    datagen = ImageDataGenerator(
        horizontal_flip=True,
        vertical_flip=True,
        # featurewise_center=True,
        # featurewise_std_normalization=True,
        # zoom_range=0.1,
        rotation_range=30,
        fill_mode='nearest',
        # zca_whitening=True,
        # shear_range=0.2,
    )
    datagen.fit(train_data)
    datagen.fit(x_val)
    model.compile(loss='binary_crossentropy',
                  optimizer=optimizer,
                  metrics=['accuracy',
                           metric_specificity,
                           recall,
                           fbeta_score])
    history = model.fit_generator(datagen.flow(train_data, train_y, batch_size=args.batchsize),
                                  epochs=args.epochs,
                                  # validation_split=0.2,
                                  steps_per_epoch=int(len(train_y) / args.epochs),
                                  validation_data=datagen.flow(x_val, y_val, batch_size=args.batchsize),
                                  validation_steps=int(len(x_val) / args.epochs),
                                  callbacks=callback_list,
                                  shuffle=True,
                                  verbose=1,
                                  )
    # model.save('model/psir_fold{}.h5'.format(fold))

    return history


def train_model(args, SQ_NAME, base_dir):
    # load data
    n = args.i
    n = str(n)
    n = n.zfill(3)
    print('Num：', n)
    list_0 = glob.glob(base_dir + '/0/{}/*.jpg'.format(SQ_NAME)) + glob.glob(
        (base_dir + '/0/{}_0/*.jpg').format(SQ_NAME))
    list_1 = glob.glob(base_dir + '/1/{}/*.jpg'.format(SQ_NAME))
    test_list = []
    if args.patients == 'has':
        label = 1
        p_num = args.has_num
        test_list = glob.glob(base_dir + '/1/{}/{}*.jpg'.format(SQ_NAME, n))
        if not test_list:
            write_default(args.i, p_num, label, SQ_NAME, save_path)
            return
        print(test_list)

        list_1 = list(set(list_1).difference(set(test_list)))
        train_data, train_y = make_data(list_1, list_0)
    elif args.patients == 'none':
        label = 0
        p_num = args.no_num
        test_list = glob.glob(base_dir + '/0/{}/{}*.jpg'.format(SQ_NAME, n))
        if not test_list:
            write_default(args.i, p_num, label, SQ_NAME, save_path)
            return
        print(test_list)

        list_0 = list(set(list_0).difference(set(test_list)))
        train_data, train_y = make_data(list_1, list_0)

    print('train data:', train_data.shape)
    print(train_y.shape)

    x_train, x_val, y_train, y_val = train_test_split(train_data, train_y, test_size=0.2, random_state=42)

    # plot(history,i=fold,save_path=save_path)
    history = train(args, x_train, y_train, x_val, y_val, SQ_NAME)
    plot_one(history, args.i, save_path + 'result', label)
    print('p_num:', p_num)
    print('label:', label)
    write_result(args.i, p_num, label, SQ_NAME, save_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="T2 train model")
    parser.add_argument('-b', '--batchsize', type=int, default='16')
    parser.add_argument('-e', '--epochs', type=int, default='15')
    parser.add_argument('-op', '--optimizers', default='Adam')
    parser.add_argument('-lr', '--learnrate', default='1e-4')
    parser.add_argument('--i', type=int, default='1')
    parser.add_argument('-p', '--patients', type=str, choices=['has', 'none'], default='none')
    parser.add_argument('--no_num', type=int, default='428')
    parser.add_argument('--has_num', type=int, default='60')
    args = parser.parse_args()

    optimizer = optimizers.Adam(lr=1e-4)
    save_path = '/home/projects/python/MACE_model/t2/'
    train_model(args, SQ_NAME, data_path)
