import logging
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
logging.getLogger("tensorflow").setLevel(logging.DEBUG)
logging.basicConfig(filename='logs.log', filemode='w', level=logging.DEBUG,
                    format='%(asctime)s: %(levelname)s - %(message)s')

import keras
import numpy as np
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.models import Model
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import multi_gpu_model

from ml import residual_model
from utils import load_images_from_folder

keras.backend.set_image_data_format('channels_last')


class Facades:  # TODO: Predict
    """Class for segmentation buildings' facades using Deep Learning"""
    def __init__(self, dim=(720, 1080), n_channels=3, use_datagenerator=True):
        """Initialize main parameters of Facades class s
        :param dim: Images resolution in format (height, width)
        :param n_channels: Number of channels in image (1 or 3)
        """
        self.batch_size = None
        self.dim = dim
        self.shape = dim + (n_channels,)
        self.n_channels = n_channels

        self.model = None
        self.training_init_check = False
        self.use_datagenerator = use_datagenerator

        if use_datagenerator:
            self.train_generator, self.valid_generator = None, None
        else:
            self.train_data, self.valid_data = None, None

    # noinspection PyAttributeOutsideInit
    def training_init(self, dataset_dir="dataset", batch_size=8):
        """Initialize the parameters and generators
        :param dataset_dir: File path to dataset directory. With folders system:
            > train
                > data
                > masks
            > valid
                > data
                > masks
        :param batch_size: Number of images in one training iteration
        """
        # TODO: os.path.join
        self.training_init_check = True
        self.batch_size = batch_size

        if self.use_datagenerator:
            # Initialize ImageDataGenerator classes
            data_datagen = ImageDataGenerator(
                rescale=1. / 255,

                height_shift_range=0.2,
                horizontal_flip=True,
                rotation_range=45,
                width_shift_range=0.2,
                zoom_range=[0.5, 1.0]
            )

            masks_datagen = ImageDataGenerator(
                height_shift_range=0.2,
                horizontal_flip=True,
                rotation_range=45,
                width_shift_range=0.2,
                zoom_range=[0.5, 1.0]
            )

            # Initialize data generators
            seed = 909  # to transform image and masks with same augmentation parameter.
            image_generator = data_datagen.flow_from_directory(dataset_dir + "train/data/", class_mode=None,
                                                               seed=seed, batch_size=batch_size,
                                                               target_size=self.dim)
            masks_generator = masks_datagen.flow_from_directory(dataset_dir + "train/masks/", class_mode=None,
                                                                seed=seed, batch_size=batch_size,
                                                                target_size=self.dim, color_mode="grayscale")

            self.train_generator = zip(image_generator, masks_generator)

            if os.path.exists(dataset_dir + "/valid"):
                valid_datagen = ImageDataGenerator(rescale=1. / 255)

                valid_image_generator = valid_datagen.flow_from_directory(dataset_dir + "/valid/masks/",
                                                                          class_mode=None, seed=seed)
                valid_masks_generator = valid_datagen.flow_from_directory(dataset_dir + "/valid/masks/",
                                                                          class_mode=None, seed=seed)

                self.valid_generator = zip(valid_image_generator, valid_masks_generator)
        else:
            x_train = load_images_from_folder(dataset_dir + "/train/data/img", self.shape)
            y_train = load_images_from_folder(dataset_dir + "/train/masks/img", self.dim + (1,))

            self.train_data = (x_train, y_train)

            if os.path.exists(dataset_dir + "/valid/img"):
                x_valid = load_images_from_folder(dataset_dir + "/valid/data/img", self.shape)
                y_valid = load_images_from_folder(dataset_dir + "/valid/masks/img", self.dim + (1,))  # TODO: Add Y num_channels variable to init

                self.valid_data = (x_valid, y_valid)

        # Initialize model
        base_model = keras.applications.xception.Xception(weights='imagenet',
                                                          include_top=False,
                                                          input_shape=self.shape)

        output_layer = residual_model(base_model, dropout_rate=0.5)

        self.model = Model(base_model.input, output_layer)
        # keras.utils.plot_model(self.model, 'model.png', show_shapes=True)

    def train(self, epochs=1, number_of_GPUs=0):
        """Train model
        :param epochs: Number of epochs to train model
        :param number_of_GPUs: Number of devices to train on multiple GPUs
        """
        # Check training init before train
        if not self.training_init_check:
            print('Please Initialize training variables before training (use "Facades.training_init" function)')
            return False

        # Initialize callbacks
        callbacks = [
            EarlyStopping(monitor='accuracy', patience=10, verbose=1),
            ModelCheckpoint('checkpoints/weights.{epoch:02d}-{accuracy:.2f}.hdf5', monitor="accuracy",
                            verbose=1, save_best_only=True, save_weights_only=True),
            ReduceLROnPlateau(monitor='accuracy', factor=0.2, patience=5, min_lr=0.001, verbose=1)
        ]

        # Run on GPU if needed
        if number_of_GPUs:
            self.model = multi_gpu_model(self.model, gpus=number_of_GPUs)

        # Run model training
        self.model.compile(optimizer=Adam(), loss="binary_crossentropy", metrics=["accuracy"])

        if self.use_datagenerator:
            self.model.fit_generator(self.train_generator, steps_per_epoch=np.floor(160 / self.batch_size),
                                     epochs=epochs, callbacks=callbacks, verbose=2)
        else:
            self.model.fit(*self.train_data, self.batch_size, epochs=epochs, callbacks=callbacks, verbose=2)


if __name__ == "__main__":
    # Initialize main class
    facades = Facades((512, 512))

    # Initialize training dataset
    facades.training_init("dataset/", batch_size=5)

    # Run model training
    facades.train(10)
