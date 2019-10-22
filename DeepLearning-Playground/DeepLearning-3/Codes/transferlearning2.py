'fine tune'
from __future__ import print_function
from __future__ import absolute_import

import warnings
warnings.filterwarnings('ignore')
from keras import applications
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Model
from keras.layers import Dropout, Flatten, Dense, Concatenate, BatchNormalization, Activation ,GlobalAveragePooling2D
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, TensorBoard
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import gc


weightsOutputFile = '../ModelCheckpoints/InceptionV3.{epoch:02d}-{val_acc:.3f}.hdf5'
IM_WIDTH, IM_HEIGHT = 250,250
EPOCH = 1
BATCH_SIZE = 32
NB_IV3_LAYERS_TO_FREEZE = 172


def add_callback():
    # tensor_board=TensorBoard(log_dir='../Graph', histogram_freq=0,batch_size=32, write_graph=True, write_images=True)
    learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', patience=5, verbose=1, factor=0.5, min_lr=0.00001)
    checkpoint = ModelCheckpoint(weightsOutputFile, monitor='val_acc', verbose=1, save_best_only=True,
                                 save_weights_only=False,
                                 mode='auto', period=1)
    # early = EarlyStopping(monitor='val_acc', min_delta=0, patience=10, verbose=1, mode='auto')
    return [checkpoint, learning_rate_reduction]


def setup_to_transfer_learn(model, base_model):
  """Freeze all layers and compile the model"""
  for layer in base_model.layers:
    layer.trainable = False
    model.compile(loss='binary_crossentropy', optimizer=optimizers.Adam(), metrics=['accuracy'])


def setup_to_finetune(model):
  """
  Freeze the bottom NB_IV3_LAYERS and retrain the remaining top layers.
  note: NB_IV3_LAYERS corresponds to the top 2 inception blocks in the inceptionv3 arch
  Args:
    model: keras model
  """
  for layer in model.layers[:NB_IV3_LAYERS_TO_FREEZE]:
     layer.trainable = False
  for layer in model.layers[NB_IV3_LAYERS_TO_FREEZE:]:
     layer.trainable = True
  model.compile(loss='binary_crossentropy', optimizer=optimizers.Adam(lr=1e-4), metrics=['accuracy'])

def add_new_last_layer(base_model, nb_classes):
    x = base_model.output
    x = Flatten()(x)
    x = Dense(1024)(x)
    x = Activation('relu')(x)
    x = Dropout(0.2)(x)
    x = Dense(512)(x)
    x = Activation('selu')(x)
  
    # using sigmoid for multi-label classification
    predictions = Dense(nb_classes, activation='sigmoid')(x)
    # Creating final model
    model = Model(inputs=base_model.input, outputs=predictions)
    return model


def build_model(X_train, X_val, y_train, y_val):
    # Data Augmentation
    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        horizontal_flip=True,
        shear_range=0.2,
        zoom_range=0.2,
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2
    )

    test_datagen = ImageDataGenerator(
        rescale=1. / 255,
        horizontal_flip=True,
        shear_range=0.2,
        zoom_range=0.2,
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2
    )

    train_generator = train_datagen.flow(
        X_train, y_train)

    validation_generator = test_datagen.flow(
        X_val, y_val
    )

    base_model = applications.InceptionV3(weights="imagenet", include_top=False,
                                          input_shape=(IM_WIDTH, IM_HEIGHT, 3))

    model = add_new_last_layer(base_model, y_train.shape[1])
    # transfer learning
    setup_to_transfer_learn(model, base_model)
    # Train model
    history1 = model.fit_generator(
        train_generator,
        steps_per_epoch=nb_train_samples / BATCH_SIZE,
        epochs=EPOCH,
        shuffle=True,
        validation_data=validation_generator,
        validation_steps=nb_validation_samples / BATCH_SIZE,
        callbacks=add_callback()
    )
    # fine-tuning
    setup_to_finetune(model)
    history2 = model.fit_generator(
        train_generator,
        steps_per_epoch=nb_train_samples / BATCH_SIZE,
        epochs=EPOCH,
        shuffle=True,
        validation_data=validation_generator,
        validation_steps=nb_validation_samples / BATCH_SIZE,
        callbacks=add_callback()
    )



if __name__ == '__main__':
    train_data = pd.read_csv('../PreProcessing/train_data.csv')
    test_data = pd.read_csv('../PreProcessing/test_data.csv')

    #####
    train_array = np.load("../PreProcessing/train_images_arrays.npz")
    train_array = train_array['arr_0']
    train_img_name = train_data['img_name'].values

    #####
    test_array = np.load("../PreProcessing/test_images_array.npz")
    test_array = test_array['arr_0']
    test_img_name = test_data['img_name'].values

    # load meta-data
    train_meta = pd.read_csv('../DL3 Dataset/meta-data/train.csv')
    test_meta = pd.read_csv('../DL3 Dataset/meta-data/test.csv')

    train_attr = [(train_meta[train_meta['Image_name'] == str(img)]).iloc[0, 1:].values for img in train_img_name]
    test_attr = [(test_meta[test_meta['Image_name'] == str(img)]).iloc[0, 1:].values for img in test_img_name]

    train_attr = np.asarray(train_attr)
    test_attr = np.asarray(test_attr)

    # free up memory
    del train_data
    del test_data
    del train_meta
    del test_meta
    print('Memory Free : ', gc.collect())

    # train test split

    X_train, X_val, y_train, y_val = train_test_split(train_array, train_attr, test_size=0.2, random_state=1,
                                                      stratify=train_attr)

    nb_train_samples = len(X_train)
    nb_validation_samples = len(X_val)

    print('Train samples: {}'.format(nb_train_samples))
    print('Validation samples: {}'.format(nb_validation_samples))
    # Model preparation
    build_model(X_train, X_val, y_train, y_val)

    # save the model
