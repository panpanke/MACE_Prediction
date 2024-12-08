import os
import numpy as np
import cv2

# SQ_NAME = 'lge'

data_path = '/home/fred/data/train'
SMALLEST_DIM = 256
IMAGE_CROP_SIZE = 128
# N_FOLD = 163
Patient_has = 60
Patient_none = 428


def crop_center(img, new_size):
    y, x, c = img.shape
    (cropx, cropy) = new_size
    startx = x // 2 - (cropx // 2)
    starty = y // 2 - (cropy // 2)
    return img[starty:starty + cropy, startx:startx + cropx]


def rescale_pixel_values(img):
    img = img.astype('float32')
    # normalize to the range 0:1
    img /= 255.0
    return img


def resize(img):
    # print('Original Dimensions : ', img.shape)
    original_width = int(img.shape[1])
    original_height = int(img.shape[0])

    aspect_ratio = original_width / original_height

    if original_height < original_width:
        new_height = SMALLEST_DIM
        new_width = int(new_height * aspect_ratio)
    else:
        new_width = SMALLEST_DIM
        new_height = int(original_width / aspect_ratio)

    dim = (new_width, new_height)
    # dim = (256, 256)
    # resize image
    resized = cv2.resize(img, dim, interpolation=cv2.INTER_LINEAR)
    # print('Resized Dimensions : ', resized.shape)

    return resized


def pre_process_rgb(img):
    resized = resize(img)
    img_cropped = crop_center(resized, (IMAGE_CROP_SIZE, IMAGE_CROP_SIZE))
    img = rescale_pixel_values(img_cropped)
    return img


def select_test_num(p_num, n):
    # n = int(p_num / N_FOLD)
    num_list = [i for i in range(1, p_num + 1)]
    list_all = []
    for i in num_list:
        i = str(i)
        i = i.zfill(3)
        list_all.append(i)

    test_list = list_all[n - 1:n]
    return test_list


def make_data(list_1, list_0):
    labels_1 = []
    data_1 = np.zeros((1, IMAGE_CROP_SIZE, IMAGE_CROP_SIZE, 3))
    for file in list_1:
        img = cv2.imread(file, cv2.IMREAD_COLOR)
        img = rescale_pixel_values(img)
        new_img = np.reshape(img, (1, IMAGE_CROP_SIZE, IMAGE_CROP_SIZE, 3))
        data_1 = np.append(data_1, new_img, axis=0)
        labels_1.append(1)
    data_1 = data_1[1:, :, :, :]

    labels_0 = []
    data_0 = np.zeros((1, IMAGE_CROP_SIZE, IMAGE_CROP_SIZE, 3))
    for file in list_0:
        img = cv2.imread(file, cv2.IMREAD_COLOR)
        img = rescale_pixel_values(img)
        new_img = np.reshape(img, (1, IMAGE_CROP_SIZE, IMAGE_CROP_SIZE, 3))
        data_0 = np.append(data_0, new_img, axis=0)
        labels_0.append(0)
    data_0 = data_0[1:, :, :, :]

    data = np.concatenate((data_1, data_0), axis=0)
    labels = labels_1 + labels_0

    np.random.seed(7)
    np.random.shuffle(data)
    train_data = np.array(data)
    np.random.seed(7)
    np.random.shuffle(labels)
    y_labels = np.array(labels)
    return train_data, y_labels
