import numpy
from keras.datasets import mnist
from keras.models import Sequential
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
weightsOutputFile = 'Digit_classifier.{epoch:02d}-{val_loss:.3f}.hdf5'
# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)

# load data
(X_train, y_train), (X_test, y_test) = mnist.load_data()
print(X_train.shape)
print(X_test.shape)
# reshape to be [samples][pixels][width][height]
X_train = X_train.reshape(X_train.shape[0], 1, 28, 28).astype('float32')
X_test = X_test.reshape(X_test.shape[0], 1, 28, 28).astype('float32')

# normalize inputs from 0-255 to 0-1
X_train = X_train / 255
X_test = X_test / 255
# one hot encode outputs
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
num_classes = y_test.shape[1]

def CNN_Model():
    model = Sequential()
    
    model.add(Convolution2D(7, 5, 5,border_mode='valid', input_shape=(1, 28, 28),name='conv1_1'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Convolution2D(7, 5, 5,border_mode='valid', name='conv1_2'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D((2, 2), strides=(1, 1)))

    model.add(Convolution2D(14, 3, 3, border_mode='valid', name='conv2_1'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Convolution2D(14, 3, 3,border_mode='valid', name='conv2_2'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D((2, 2), strides=(1, 1)))

    model.add(Convolution2D(28, 1, 1,border_mode='valid', name='conv3_1'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Convolution2D(28, 1, 1,border_mode='valid', name='conv3_2'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D((2, 2), strides=(1, 1)))


    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(112))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dense(56))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dense(10))
    model.add(BatchNormalization())
    model.add(Activation('softmax'))
    # Compile model
    model.compile(loss='categorical_crossentropy',optimizer='adam', metrics=['accuracy'])
    return model 


model=CNN_Model()
checkpointer = ModelCheckpoint(weightsOutputFile, monitor='accuracy', save_best_only=False, mode='auto')
model.fit(X_train, y_train, batch_size=100, nb_epoch=20,verbose=1, validation_data=(X_test, y_test),callbacks=[checkpointer])
scores = model.evaluate(X_test, y_test, verbose=1)
print("Baseline Error: %.2f%%" % (100 - scores[1] * 100))
