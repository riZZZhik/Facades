import os

import numpy as np
from PIL import Image
from tqdm import tqdm


# def fix_dataset_labels(labels_path, colors=None):
#     if colors is None:
#         colors = {  # RGB
#             (0, 0, 255): 0,  # Wall
#             (0, 0, 170): 1,  # Background
#             (255, 255, 0): 2,  # Window (closed)
#             (0, 85, 255): 2,  # Window
#             (0, 170, 255): 7,  # Door
#             (170, 0, 0): 8,  # Shop
#             (170, 255, 85): 3,  # Balcony
#             (255, 85, 0): 6,  # Molding
#             (255, 3, 0): 11,  # Pillar
#             (0, 255, 255): 12,  # Cornice
#             (85, 255, 170): 4,  # Sill
#             (255, 170, 0): 9,  # Deco
#             # (): 11,  # Blind
#         }
#
#         try:
#             os.mkdir(labels_path + "_FIXED/")
#         except:
#             pass
#
#         files = [f for f in os.listdir(labels_path) if f[-4:] in [".jpg", ".png"]]
#         for id, image_name in enumerate(tqdm(files)):
#             image_path = labels_path + "/" + image_name
#             image = cv.imread(image_path)
#
#             for h in range(len(image)):
#                 for w in range(len(image[h])):
#                     c = image[h][w]
#                     for key, value in colors.items():
#                         if c[2] in range(key[0] - 3, key[0] + 3) and c[1] in range(key[1] - 2, key[1] + 3) and \
#                                 c[0] in range(key[2] - 2, key[2] + 3):
#                             image[h][w] = value
#                             break
#
#             image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
#             cv.imwrite(labels_path + "_FIXED/" + image_name, image)


# def show_batch_photos(facades_class):
#     x_batch, y_batch = next(facades_class.train_generator)
#     for i in range(0, facades_class.batch_size):
#         x = x_batch[i]
#         y = y_batch[i]
#
#         plt.imshow(x)
#         plt.show()  # TODO: Check needing two
#         plt.imshow(y)
#         plt.show()


def load_images_from_folder(dir, shape):
    files = [file for file in os.listdir(dir) if file.endswith((".png", ".jpg"))]
    result = []

    for image_name in tqdm(files, dir):
        image_path = dir + "/" + image_name
        image = Image.open(image_path)
        image_resized = image.resize(shape[:-1], Image.ANTIALIAS)
        image_numpy = np.array(image_resized)
        if shape[-1] == 1:
            image_numpy = np.expand_dims(image_numpy, axis=-1)

        result.append(image_numpy)

    return np.array(result)


