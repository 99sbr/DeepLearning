import gc
gc.collect()
from keras.models import Sequential, Model
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten, Activation
from keras.layers.convolutional import Convolution2D, SeparableConvolution2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.pooling import GlobalAveragePooling2D, AveragePooling2D
from keras.layers import Input, merge
from keras.utils import np_utils
from keras import backend as K
K.set_image_dim_ordering('tf')

nb_filters_reduction_factor = 8


def Stem(Input):
    print(Input)
    x = Convolution2D(32 // nb_filters_reduction_factor,
                      3, 3, subsample=(1, 1))(Input)
    x = Convolution2D(32 // nb_filters_reduction_factor, 3, 3)(x) 
    x = Convolution2D(32 // nb_filters_reduction_factor, 3, 3)(x) 
    x = Convolution2D(64 // nb_filters_reduction_factor,
                      3, 3, border_mode='same')(x)

    path1 = MaxPooling2D((3, 3), strides=(1, 1))(x)  # changed 
    path2 = Convolution2D(96 // nb_filters_reduction_factor,
                          3, 3, subsample=(1, 1))(x) #changed
    y = merge([path1, path2], mode='concat')

    a = Convolution2D(64 // nb_filters_reduction_factor,
                      1, 1, border_mode='same')(y)
    a = Convolution2D(64 // nb_filters_reduction_factor,
                      3, 1, border_mode='same')(a)
    a = Convolution2D(64 // nb_filters_reduction_factor,
                      1, 3, border_mode='same')(a)
    a = Convolution2D(96 // nb_filters_reduction_factor,
                      3, 3, border_mode='valid')(a)

    b = Convolution2D(64 // nb_filters_reduction_factor,
                      1, 1, border_mode='same')(y)
    b = Convolution2D(96 // nb_filters_reduction_factor,
                      3, 3, border_mode='valid')(b)

    z = merge([a, b], mode='concat')
    z1 = MaxPooling2D((3, 3), strides=(1, 1))(z)
    z2 = Convolution2D(192 // nb_filters_reduction_factor, 3,
                       3, subsample=(1, 1), border_mode='valid')(z)

    c = merge([z1, z2], mode='concat', concat_axis=-1)
    return c


def Inception_A(Input):
    path1 = Convolution2D(64 // nb_filters_reduction_factor,
                          1, 1, border_mode='same', activation='relu')(Input)
    path1 = Convolution2D(96 // nb_filters_reduction_factor,
                          3, 3, border_mode='same', activation='relu')(path1)
    path1 = Convolution2D(96 // nb_filters_reduction_factor,
                          3, 3, border_mode='same', activation='relu')(path1)

    path2 = Convolution2D(64 // nb_filters_reduction_factor,
                          1, 1, border_mode='same', activation='relu')(Input)
    path2 = Convolution2D(96 // nb_filters_reduction_factor,
                          3, 3, border_mode='same', activation='relu')(path2)

    path3 = Convolution2D(96 // nb_filters_reduction_factor,
                          1, 1, border_mode='same', activation='relu')(Input)

    path4 = AveragePooling2D((3, 3), strides=(1, 1), border_mode='same')(Input)
    path4 = Convolution2D(96 // nb_filters_reduction_factor,
                          1, 1, border_mode='same', activation='relu')(path4)

    output = merge([path1, path2, path3, path4], mode='concat', concat_axis=-1)
    return output


def Reduction_A(Input):
    path1 = Convolution2D(k // nb_filters_reduction_factor,
                          1, 1, border_mode='same', activation='relu')(Input)
    path1 = Convolution2D(l // nb_filters_reduction_factor,
                          3, 3, border_mode='same', activation='relu')(path1)
    path1 = Convolution2D(m // nb_filters_reduction_factor, 3, 3,
                          subsample=(2, 2), border_mode='valid', activation='relu')(path1)

    path2 = Convolution2D(n // nb_filters_reduction_factor, 3, 3,
                          subsample=(2, 2), border_mode='valid', activation='relu')(Input)

    path3 = MaxPooling2D((3, 3), strides=(2, 2), border_mode='valid')(Input)

    output = merge([path1, path2, path3], mode='concat', concat_axis=-1)
    return output


def Inception_B(Input):
    path1 = Convolution2D(192 // nb_filters_reduction_factor,
                          1, 1, border_mode='same', activation='relu')(Input)
    path1 = Convolution2D(192 // nb_filters_reduction_factor,
                          1, 7, border_mode='same', activation='relu')(path1)
    path1 = Convolution2D(224 // nb_filters_reduction_factor,
                          7, 1, border_mode='same', activation='relu')(path1)
    path1 = Convolution2D(224 // nb_filters_reduction_factor,
                          1, 7, border_mode='same', activation='relu')(path1)
    path1 = Convolution2D(256 // nb_filters_reduction_factor,
                          7, 1, border_mode='same', activation='relu')(path1)

    path2 = Convolution2D(192 // nb_filters_reduction_factor,
                          1, 1, border_mode='same', activation='relu')(Input)
    path2 = Convolution2D(223 // nb_filters_reduction_factor,
                          1, 7, border_mode='same', activation='relu')(path2)
    path2 = Convolution2D(256 // nb_filters_reduction_factor,
                          1, 7, border_mode='same', activation='relu')(path2)

    path3 = Convolution2D(384 // nb_filters_reduction_factor,
                          1, 1, border_mode='same', activation='relu')(Input)

    path4 = AveragePooling2D((3, 3), strides=(1, 1), border_mode='same')(Input)
    path4 = Convolution2D(128 // nb_filters_reduction_factor,
                          1, 1, border_mode='same', activation='relu')(path4)

    output = merge([path1, path2, path3, path4], mode='concat', concat_axis=-1)

    return output


def Reduction_B(Input):
    path1 = Convolution2D(256 // nb_filters_reduction_factor,
                          1, 1, border_mode='same', activation='relu')(Input)
    path1 = Convolution2D(256 // nb_filters_reduction_factor,
                          1, 7, border_mode='same', activation='relu')(path1)
    path1 = Convolution2D(320 // nb_filters_reduction_factor,
                          7, 1, border_mode='same', activation='relu')(path1)
    path1 = Convolution2D(320 // nb_filters_reduction_factor, 3, 3,
                          subsample=(2, 2), border_mode='valid', activation='relu')(path1)

    path2 = Convolution2D(192 // nb_filters_reduction_factor,
                          1, 1, border_mode="same", activation='relu')(Input)
    path2 = Convolution2D(192 // nb_filters_reduction_factor, 3, 3,
                          subsample=(2, 2), border_mode="valid", activation='relu')(path2)

    path3 = MaxPooling2D((3, 3), strides=(2, 2))(Input)

    output = merge([path1, path2, path3], mode='concat', concat_axis=-1)
    return output


def Inception_C(Input):
    path1 = Convolution2D(384 // nb_filters_reduction_factor,
                          1, 1, border_mode='same', activation='relu')(Input)
    path1 = Convolution2D(448 // nb_filters_reduction_factor,
                          1, 3, border_mode='same', activation='relu')(path1)
    path1 = Convolution2D(512 // nb_filters_reduction_factor,
                          3, 1, border_mode='same', activation='relu')(path1)
    path1_a = Convolution2D(256 // nb_filters_reduction_factor,
                            1, 3, border_mode='same', activation='relu')(path1)
    path1_b = Convolution2D(256 // nb_filters_reduction_factor,
                            3, 1, border_mode='same', activation='relu')(path1)

    path2 = Convolution2D(384 // nb_filters_reduction_factor,
                          1, 1, border_mode='same', activation='relu')(Input)
    path2_a = Convolution2D(256 // nb_filters_reduction_factor,
                            3, 1, border_mode='same', activation='relu')(path2)
    path2_b = Convolution2D(256 // nb_filters_reduction_factor,
                            1, 3, border_mode='same', activation='relu')(path2)

    path3 = Convolution2D(256 // nb_filters_reduction_factor,
                          1, 1, border_mode='same', activation='relu')(Input)

    path4 = AveragePooling2D((3, 3), strides=(1, 1), border_mode='same')(Input)
    path4 = Convolution2D(256 // nb_filters_reduction_factor,
                          1, 1, border_mode='same', activation='relu')(Input)

    output = merge([path1, path2, path3, path4], mode='concat', concat_axis=-1)
    return output



img_rows, img_cols = 28, 28
img_channels = 1
k = 192
l = 224
m = 256
n = 384
# in original inception-v4, these are 4, 7, 3, respectively
num_A_blocks = 1
num_B_blocks = 1
num_C_blocks = 1
nb_classes = 10
inputs = Input(shape=(img_rows, img_cols,img_channels))
x = Stem(inputs)
for i in range(num_A_blocks):
    x = Inception_A(x)
x = Reduction_A(x)
for i in range(num_B_blocks):
    x = Inception_B(x)
x = Reduction_B(x)
for i in range(num_C_blocks):
    x = Inception_C(x)

x = AveragePooling2D(pool_size=(3, 3), strides=(
    1, 1), border_mode='valid', dim_ordering='tf')(x)
x = Dropout(0.5)(x)
print(x.get_shape())
x = Flatten()(x)
print(x.get_shape())
predictions = Dense(nb_classes, activation='softmax')(x)
model = Model(input=inputs, output=predictions)
print(model.summary())
