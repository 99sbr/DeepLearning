import albumentations
import pandas as pd
import json
import numpy as np
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.platform import flags
from PIL import Image


FLAGS = flags.FLAGS
augmentations = albumentations.Compose([
    albumentations.Transpose(p=0.5),
    albumentations.HorizontalFlip(p=0.5),
    albumentations.VerticalFlip(p=0.5),
    albumentations.ColorJitter(p=0.5),
    albumentations.ShiftScaleRotate(p=0.5),
    albumentations.HueSaturationValue(
        hue_shift_limit=0.2,
        sat_shift_limit=0.2,
        val_shift_limit=0.2,
        p=0.5
    ),
    albumentations.RandomBrightnessContrast(
        brightness_limit=(-0.1, 0.1),
        contrast_limit=(-0.1, 0.1),
        p=0.5
    ),
    albumentations.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
        max_pixel_value=255.0,
        p=1.0
    ),
    albumentations.CoarseDropout(p=0.5)], p=1.)


def aug_fn(image):
    data = {"image": image.astype(np.uint8)}
    aug_data = augmentations(**data)
    aug_img = aug_data["image"]
    aug_img = tf.cast(aug_img/255.0, tf.float32)
    return aug_img


def get_train_and_valid_data():
    labeled_datagen = ImageDataGenerator(validation_split=FLAGS.validation_split,
                                         preprocessing_function=aug_fn)

    dataframe = pd.read_csv(FLAGS.dataframe)
    dataframe['image_id'] = dataframe['image_id'].apply(lambda x: x+'.jpg')
    train_dataset = labeled_datagen.flow_from_dataframe(dataframe,
                                                        directory=FLAGS.img_dir,
                                                        x_col='image_id',
                                                        y_col='breed',
                                                        target_size=(
                                                            FLAGS.img_width, FLAGS.img_height),
                                                        batch_size=FLAGS.batch_size,
                                                        shuffle=True,
                                                        color_mode='rgb',
                                                        class_mode='categorical',
                                                        subset='training',
                                                        interpolation='bicubic')
    valid_dataset = labeled_datagen.flow_from_dataframe(dataframe,
                                                        directory=FLAGS.img_dir,
                                                        x_col='image_id',
                                                        y_col='breed',
                                                        target_size=(
                                                            FLAGS.img_width, FLAGS.img_height),
                                                        batch_size=FLAGS.batch_size,
                                                        shuffle=True,
                                                        color_mode='rgb',
                                                        class_mode='categorical',
                                                        subset='validation',
                                                        interpolation='bicubic')

    return train_dataset, valid_dataset
