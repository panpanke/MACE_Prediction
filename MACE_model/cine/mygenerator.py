import cv2
import numpy as np
import keras


class DataGenerator(keras.utils.Sequence):
    """Generates data for Keras"""

    def __init__(self, list_IDs, labels, img_dir, batch_size=16, img_size=(30, 128, 128), n_channels=1,
                 n_classes=2, shuffle=True):
        """Initialization"""
        self.img_size = img_size
        self.batch_size = batch_size
        self.labels = labels
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()
        self.imgdir = img_dir

    def __len__(self):
        """Denotes the number of batches per epoch"""
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        """Generate one batch of data"""
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, Y = self.__data_generation(list_IDs_temp)

        return X, Y

    def on_epoch_end(self):
        """Updates indexes after each epoch"""
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        """Generates data containing batch_size samples"""  # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.img_size, self.n_channels))
        Y = np.empty((self.batch_size), dtype=int)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            X[i,] = np.load('data/' + ID + '.npy')

            # Store class
            Y[i] = self.labels[ID]

        # return X, keras.utils.to_categorical(y, num_classes=self.n_classes)
        return X, Y


class data_generator:
    def __init__(self, trainset, labels, image_size, batchsize=16, classes=2):
        self.dataset = trainset
        self.labels = labels
        self.index = 0
        self.batch_size = batchsize
        self.image_size = image_size
        self.classes = classes

    def get_mini_batch(self):
        rgb_list = []
        for file in self.dataset:
            result = np.zeros((1, 128, 128, 3))
            for img in file:
                img = cv2.imread(img, cv2.IMREAD_UNCHANGED)
                img = img.astype('float32')
                # normalize to the range 0:1
                img /= 255.0
                new_img = np.reshape(img, (1, 128, 128, 3))
                result = np.append(result, new_img, axis=0)
            result = result[1:, :, :, :]
            rgb_list.append(result)
        while True:
            batch_images = []
            batch_labels = []
            for i in range(self.batch_size):
                if (self.index == len(rgb_list)):
                    self.index = 0
                batch_images.append(rgb_list[self.index])
                batch_labels.append(self.labels[self.index])
                self.index += 1
            batch_images = np.array(batch_images)
            batch_labels = np.array(batch_labels)
            yield batch_images, batch_labels
