import os

import keras
import numpy as np
import tensorflow as tf
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.models import Model
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator

from ml.utils import residual_model

keras.backend.set_image_data_format('channels_last')


class Facades:  # TODO: Predict  # TODO: Comments
    def __init__(self, dim=(720, 1080), n_channels=3):
        """Initialize main parameters of Facades class
        :param dim: Images resolution in format (height, width)
        :param n_channels: Number of channels in image (1 or 3)
        """
        self.dim = dim
        self.n_channels = n_channels
        self.shape = dim + (n_channels,)

        self.model = None
        self.train_generator, self.valid_generator = None, None
        self.batch_size = None

        # TODO: Train init bool
        self.training_init_check = False

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
        self.training_init_check = True
        self.batch_size = batch_size

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
        image_generator = data_datagen.flow_from_directory(dataset_dir + "train/data/", class_mode=None, seed=seed,
                                                           batch_size=batch_size, target_size=self.dim)
        masks_generator = masks_datagen.flow_from_directory(dataset_dir + "train/masks/", class_mode=None, seed=seed,
                                                            color_mode="grayscale", batch_size=batch_size,
                                                            target_size=self.dim)

        self.train_generator = zip(image_generator, masks_generator)

        if os.path.exists(dataset_dir + "valid"):
            valid_datagen = ImageDataGenerator(rescale=1. / 255)

            valid_image_generator = valid_datagen.flow_from_directory(dataset_dir + "valid/masks/",
                                                                      class_mode=None, seed=seed)
            valid_masks_generator = valid_datagen.flow_from_directory(dataset_dir + "valid/masks/",
                                                                      class_mode=None, seed=seed)

            self.valid_generator = zip(valid_image_generator, valid_masks_generator)

        # Initialize model
        base_model = keras.applications.xception.Xception(weights='imagenet',
                                                          include_top=False,
                                                          input_shape=self.shape)

        output_layer = residual_model(base_model, dropout_rate=0.5)

        self.model = Model(base_model.input, output_layer)
        self.model.compile(optimizer=Adam(), loss="binary_crossentropy", metrics=["accuracy"])
        # keras.utils.plot_model(self.model, 'model.png', show_shapes=True)

    def train(self, epochs=1, use_gpu=True):
        """Train model
        :param epochs: Number of epochs to train model
        :param use_gpu: Whether to use GPU or CPU
        """
        # Check training init before train
        if not self.training_init_check:
            print('Please Initialize training variables before training (use "Facades.training_init" function)')
            return False

        # Setup model training device
        if use_gpu:
            device = None  # TODO: Setup GPU training
        else:
            os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
            os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
            device = '/cpu:0'

        # Initialize callbacks
        callbacks = [
            EarlyStopping(patience=10, verbose=1),
            ModelCheckpoint('checkpoints/weights.{epoch:02d}-{val_loss:.2f}.hdf5', verbose=1,
                            save_best_only=True, save_weights_only=True),
            ReduceLROnPlateau(factor=0.1, patience=3, min_lr=0.00001, verbose=1)
        ]

        # Run model training
        with tf.device(device):
            self.model.fit_generator(self.train_generator, steps_per_epoch=np.floor(160 / self.batch_size),
                                     epochs=epochs, callbacks=callbacks, verbose=2)


if __name__ == "__main__":
    facades = Facades((1024, 512))
