import os

import keras

from ml.utils import residual_model

keras.backend.set_image_data_format('channels_last')

DATASET_DIR = "dataset"


class Facades:  # TODO: Predict  # TODO: Comments
    def __init__(self, dim=(720, 1080), n_channels=3):
        self.dim = dim
        self.n_channels = n_channels
        self.shape = dim + (n_channels,)

        self.model = None
        self.train_generator, self.valid_generator = None, None
        self.batch_size = None

        # TODO: Train init bool
        self.training_init_check = False

    # noinspection PyAttributeOutsideInit
    def training_init(self, dataset_dir=DATASET_DIR, batch_size=8, dropout_rate=0.5):  # TODO: Desribe params
        """Initialize the parameters and generators

        :param dataset_dir: File path to dataset directory. With folders system:
            > train
                > data
                > masks
            > valid
                > data
                > masks
        """
        self.batch_size = batch_size

        # Initialize ImageDataGenerator classes
        data_datagen = ImageDataGenerator(
            rescale=1. / 255,

            brightness_range=[0.5, 1.4],
            height_shift_range=0.2,
            horizontal_flip=True,
            rotation_range=45,
            width_shift_range=0.2,
            zoom_range=[0.5, 1.0]
        )

        masks_datagen = ImageDataGenerator(
            brightness_range=[0.5, 1.4],
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

        output_layer = residual_model(base_model, dropout_rate)

        self.model = Model(base_model.input, output_layer)
        self.model.compile(optimizer=Adam(), loss="binary_crossentropy", metrics=["accuracy"])
        # keras.utils.plot_model(self.model, 'model.png', show_shapes=True)

if __name__ == "__main__":
    facades = Facades((512, 1024))

    facades.training_init("dataset/", batch_size=8)
