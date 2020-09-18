import numpy as np
from cv2 import imread
import os
from keras.utils import Sequence


class Generator(Sequence):
    """Generates data for Keras"""

    def __init__(self, dataset_dir, batch_size=16, dim=(1920, 1080), n_channels=3, shuffle=True):
        """Initialization"""
        data_folder = dataset_dir + "/data"
        self.data_paths = os.listdir(data_folder)

        labels_path = dataset_dir + "/masks"
        self.labels_paths = os.listdir(labels_path)

        self.batch_size = batch_size
        self.dim = dim
        self.n_channels = n_channels

        self.shuffle = shuffle

        self.len = len(self.data_paths)

    def __len__(self):
        """Denotes the number of batches per epoch
        :return: number of batches per epoch
        """
        return len(np.floor(self.len / self.batch_size))

    def __getitem__(self, index):
        """Generate one batch of data
        :param index: index of the batch
        :return: X and y when fitting. X only when predicting
        """
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        # Download data by indexes
        X, y = [], []
        for i in indexes:
            X.append(imread(self.data_paths[i]))
            y.append(get_label(self.labels_paths[i]))

        return X, y

    def on_epoch_end(self):
        """Updates indexes after each epoch"""
        self.indexes = np.arange(self.len)
        if self.shuffle:
            np.random.shuffle(self.indexes)
