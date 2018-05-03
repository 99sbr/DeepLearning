from keras.datasets import mnist
from keras.models import Sequential,Model
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten,Activation
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers import Input 
from keras.utils import np_utils
from keras import backend as K
from keras.layers.normalization import BatchNormalization
K.set_image_dim_ordering('th')
from keras.callbacks import ModelCheckpoint
from keras.callbacks import LearningRateScheduler
import pandas as pd
import numpy as np
import math
from keras.preprocessing.image import ImageDataGenerator
from xception import  Xception_Module
# fix random seed for reproducibility
seed = 126
batch_size=32
nb_epoch =10
np.random.seed(seed)
weightsOutputFile = 'Digit_classifier.{epoch:02d}-{val_loss:.3f}.hdf5'

train=pd.read_csv('train.csv')


train_labels=train.label
train=train.drop('label',1)
train_mat=np.zeros((train.shape[0],28,28))


for i in range(len(train)):
    arr=np.asarray(train.iloc[i])
    train_mat[i]=arr.reshape(28,28)

split_index=int(len(train_labels)*(0.8))
X_train=train_mat[:split_index]
X_test=train_mat[split_index:]
Y_train=train_labels[:split_index]
Y_test=train_labels[split_index:]
X_train = X_train.reshape(X_train.shape[0], 1, 28, 28).astype('float32')
X_test = X_test.reshape(X_test.shape[0], 1, 28, 28).astype('float32')
# one hot encode outputs
Y_train = np_utils.to_categorical(Y_train)
Y_test = np_utils.to_categorical(Y_test)
num_classes = Y_train.shape[1]

model=Xception_Module()
checkpointer = ModelCheckpoint(weightsOutputFile, monitor='accuracy', save_best_only=False, mode='auto')
model.fit(X_train, Y_train, batch_size=32, nb_epoch=10,verbose=1, validation_data=(X_test,Y_test),callbacks=[checkpointer])




