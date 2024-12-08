import glob
import os

import cv2
from keras.models import load_model
import csv
from util.keras_metrics import *

SMALLEST_DIM = 256
IMAGE_CROP_SIZE = 128
FRAME_RATE = 30
SQ_NAME = 'cine'
base_dir = '/home/projects/python/medical_imaging/cine/'
data_dir = '/home/data/train'


def load_data(imglist):
    print('[INFO] loading data ...')
    test_x = []
    for i in range(int(len(imglist) / 30)):
        test_x.append(imglist[i * 30:i * 30 + 30])
    print(test_x)
    test_data = []
    for file in test_x:
        result = np.zeros((1, 128, 128, 3))
        for img in file:
            img = cv2.imread(img, cv2.IMREAD_UNCHANGED)
            img = img.astype('float32')
            # normalize to the range 0:1
            img /= 255.0
            new_img = np.reshape(img, (1, 128, 128, 3))
            result = np.append(result, new_img, axis=0)
        result = result[1:, :, :, :]
        test_data.append(result)
    test_data = np.array(test_data)
    print(test_data.shape)
    return test_data


def predict(i, model, label):
    imgs = glob.glob(r'{}/{}/{}/{}*.jpg'.format(data_dir, label, SQ_NAME, i))
    print(imgs)
    if not imgs:
        acc = -1
    else:
        test_data = load_data(imgs)
        # print(train_x.shape)
        acc = model.predict(test_data)
        acc = np.mean(acc)
        print('acc: ', acc)
    return i, acc, label


def select_testNum(p_num, n):
    num_list = [i for i in range(1, p_num + 1)]
    list_all = []
    for i in num_list:
        i = str(i)
        i = i.zfill(3)
        list_all.append(i)

    test_list = list_all[n - 1:n]
    return test_list


def write_result(fold, P_NUM, LABEL, SQ_NAME):
    model_path = base_dir + 'model/{}_model.h5'.format(SQ_NAME)

    model = load_model(model_path, custom_objects={'recall': recall,
                                                   'metric_specificity': metric_specificity,
                                                   'fbeta_score': fbeta_score})
    test_listNum = select_testNum(P_NUM, fold)
    print(test_listNum)
    print('starting predict ...')
    f = open(base_dir + 'result/{}_fold{}_{}.csv'.format(SQ_NAME, fold, LABEL), 'w', encoding='utf-8', newline='')
    csv_writer = csv.writer(f)
    csv_writer.writerow(["Num", "{}_prediction".format(SQ_NAME), "label"])

    # p_num = 163
    for i in test_listNum:
        number, acc, label = predict(i, model, LABEL)

        csv_writer.writerow([number, acc, label])
    f.close()


def write_default(fold, P_NUM, LABEL, SQ_NAME):
    test_list = select_testNum(P_NUM, fold)
    print(test_list)
    print('Passing predict ...')
    f = open(base_dir + 'result/{}_fold{}_{}.csv'.format(SQ_NAME, fold, LABEL), 'w', encoding='utf-8', newline='')
    csv_writer = csv.writer(f)
    csv_writer.writerow(["Num", "{}_prediction".format(SQ_NAME), "label"])

    # p_num = 163
    for i in test_list:
        number, acc, label = i, -1, LABEL
        csv_writer.writerow([number, acc, label])
    f.close()


def test(data_dir, SQ_NAME, TestPart, P_NUM):
    model_path = './model/{}_model.h5'.format(SQ_NAME)

    model = load_model(model_path, custom_objects={'recall': recall,
                                                   'metric_specificity': metric_specificity,
                                                   'fbeta_score': fbeta_score})
    if TestPart == "internal":
        test_listNum = [str(i).zfill(3) for i in range(1, P_NUM + 1)]
    elif TestPart == 'external':
        test_listNum = extract_image_numbers(data_dir + '/{}'.format(SQ_NAME), 4)
    else:
        test_listNum = extract_image_numbers(data_dir + '/{}'.format(SQ_NAME), 3)

    print(test_listNum)
    print('starting predict ...')
    f = open('./result/test/{}_{}.csv'.format(TestPart, SQ_NAME), 'a', encoding='utf-8', newline='')
    csv_writer = csv.writer(f)
    csv_writer.writerow(["Num", "{}_prediction".format(SQ_NAME), "label"])
    for i in test_listNum:
        imgs = glob.glob(r'{}/{}/{}*.jpg'.format(data_dir, SQ_NAME, i))
        print(imgs)
        if not imgs:
            acc = -1
        else:
            test_data = load_data(imgs)
            acc = model.predict(test_data)
            acc = np.mean(acc)
            print('acc: ', acc)
        label = 0
        csv_writer.writerow([i, acc, label])
    f.close()


def extract_image_numbers(folder_path, n):
    """
    Extract image numbers from filenames in a folder and return a sorted list of unique numbers.

    Args:
      folder_path: The path to the folder containing the images.

    Returns:
      A sorted list of unique image numbers.
    """

    image_numbers = []
    for filename in os.listdir(folder_path):
        if filename.endswith('.jpg'):
            image_number = filename[:n]
            image_numbers.append(image_number)

    # Remove duplicates and sort the list
    image_numbers = set(image_numbers)
    image_numbers = sorted(image_numbers)
    print(image_numbers)
    return image_numbers


if __name__ == '__main__':
    P_NUM = 123
    TestPart = 'test0302'
    path = r'\data\Test0302-train_extern\cut'
    test(path, SQ_NAME, TestPart, P_NUM)
