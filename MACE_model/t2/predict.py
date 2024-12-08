import glob
import os

import cv2
import numpy as np
from keras.models import load_model
import csv

from util.keras_metrics import metric_specificity, recall, fbeta_score
from util.load_data import select_test_num, data_path

SQ_NAME = 't2'
SMALLEST_DIM = 256
IMAGE_CROP_SIZE = 128
FRAME_RATE = 30


def rescale_pixel_values(img):
    img = img.astype('float32')
    # normalize to the range 0:1
    img /= 255.0
    return img


def run_imgs(sorted_list_frames, label):
    labels = []
    result = np.zeros((1, IMAGE_CROP_SIZE, IMAGE_CROP_SIZE, 3))
    for file in sorted_list_frames:
        img = cv2.imread(file, cv2.IMREAD_COLOR)
        img = rescale_pixel_values(img)
        new_img = np.reshape(img, (1, IMAGE_CROP_SIZE, IMAGE_CROP_SIZE, 3))
        result = np.append(result, new_img, axis=0)
    result = result[1:, :, :, :]
    labels.append(label)
    return result, labels


# model_path = 'model/psir_model.h5'


def predict(i, model, label):
    # n = str(i).zfill(3)
    imgs = glob.glob(r'{}/{}/{}/{}*.jpg'.format(data_path, label, SQ_NAME, i))
    print(imgs)
    if not imgs:
        acc = -1
    else:
        imgs_list, labels = run_imgs(imgs, label)
        train_x = np.array(imgs_list)
        train_y = np.array(labels)
        # print(train_x.shape)
        acc = model.predict(train_x)
        acc = np.mean(acc)
        print(acc)
    return i, acc, label


def write_result(fold, P_NUM, LABEL, SQ_NAME, base_dir):
    model_path = base_dir + 'model/{}_model.h5'.format(SQ_NAME)

    model = load_model(model_path, custom_objects={'recall': recall,
                                                   'metric_specificity': metric_specificity,
                                                   'fbeta_score': fbeta_score})
    test_listNum = select_test_num(P_NUM, fold)
    print(test_listNum)
    print('starting predict ...')
    f = open(base_dir + 'result/{}/{}_fold{}_{}.csv'.format(str(LABEL), SQ_NAME, fold, LABEL), 'w',
             encoding='utf-8', newline='')
    csv_writer = csv.writer(f)
    csv_writer.writerow(["Num", "{}_prediction".format(SQ_NAME), "label"])

    for i in test_listNum:
        number, acc, label = predict(i, model, LABEL)

        csv_writer.writerow([number, acc, label])
    f.close()


def write_default(fold, P_NUM, LABEL, SQ_NAME, base_dir):
    test_list = select_test_num(P_NUM, fold)
    print(test_list)
    print('Passing predict ...')
    f = open(base_dir + 'result/{}/{}_fold{}_{}.csv'.format(SQ_NAME + str(LABEL), SQ_NAME, fold, LABEL), 'w',
             encoding='utf-8', newline='')
    csv_writer = csv.writer(f)
    csv_writer.writerow(["Num", "{}_prediction".format(SQ_NAME), "label"])

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
    # f = open('./result/test/{}_{}.csv'.format(TestPart,SQ_NAME), 'a', encoding='utf-8', newline='')
    # csv_writer = csv.writer(f)
    # csv_writer.writerow(["Num", "{}_prediction".format(SQ_NAME), "label"])
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
    #     csv_writer.writerow([i, acc, label])
    # f.close()


def load_data(sorted_list_frames):
    result = np.zeros((1, IMAGE_CROP_SIZE, IMAGE_CROP_SIZE, 3))
    for file in sorted_list_frames:
        img = cv2.imread(file, cv2.IMREAD_COLOR)
        img = rescale_pixel_values(img)
        new_img = np.reshape(img, (1, IMAGE_CROP_SIZE, IMAGE_CROP_SIZE, 3))
        result = np.append(result, new_img, axis=0)
    result = result[1:, :, :, :]
    return result


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
    # write_result(SQ_NAME)
    test_part = 'internal'
    P_NUM = 123
    path = '/data/{}'.format(test_part)
    test(path, SQ_NAME, test_part, P_NUM)
