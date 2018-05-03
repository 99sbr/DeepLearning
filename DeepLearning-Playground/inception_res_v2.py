import gc
gc.collect()
from keras.layers.normalization import BatchNormalization
from keras.models import Sequential, Model
from keras.layers import Dense,Lambda
from keras.layers import Dropout
from keras.layers import Flatten, Activation
from keras.layers.convolutional import Convolution2D, SeparableConvolution2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.pooling import GlobalAveragePooling2D, AveragePooling2D
from keras.layers import Input, merge
from keras.utils import np_utils
from keras import backend as K
from keras.utils.layer_utils import layer_from_config
K.set_image_dim_ordering('tf')

nb_filters_reduction_factor = 8
channel_axis=-1


def Stem(Input):
    print(Input)
    x = Convolution2D(32 // nb_filters_reduction_factor,
                      3, 3, subsample=(2, 2))(Input)
    x = Convolution2D(32 // nb_filters_reduction_factor, 3, 3)(x) 
    x = Convolution2D(32 // nb_filters_reduction_factor, 3, 3)(x) 
    x = Convolution2D(64 // nb_filters_reduction_factor,
                      3, 3, border_mode='same')(x)

    path1 = MaxPooling2D((3, 3), strides=(2, 2))(x)  # changed 
    path2 = Convolution2D(96 // nb_filters_reduction_factor,
                          3, 3, subsample=(2, 2))(x) #changed
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
    z1 = MaxPooling2D((3, 3), strides=(2, 2))(z)
    z2 = Convolution2D(192 // nb_filters_reduction_factor, 3,
                       3, subsample=(2, 2), border_mode='valid')(z)

    c = merge([z1, z2], mode='concat', concat_axis=-1)
    c = BatchNormalization(axis=channel_axis)(c)
    return c


def Inception_A(Input):
  Input=Activation('relu')(Input)
  path1=Convolution2D(32//nb_filters_reduction_factor,1,1,border_mode='same',activation='relu')(Input)
  path1=Convolution2D(48//nb_filters_reduction_factor,3,3,border_mode='same',activation='relu')(path1)
  path1=Convolution2D(64//nb_filters_reduction_factor,3,3,border_mode='same',activation='relu')(path1)

  path2=Convolution2D(32//nb_filters_reduction_factor,1,1,border_mode='same',activation='relu')(Input)
  path2=Convolution2D(32//nb_filters_reduction_factor,3,3,border_mode='same',activation='relu')(path2)

  path3=Convolution2D(32//nb_filters_reduction_factor,1,1,border_mode='same',activation='relu')(Input)

  out=merge([path3,path1,path2],mode='concat',concat_axis=-1)
  out=Convolution2D(384//nb_filters_reduction_factor,1,1,border_mode='same')(out)
  out = Lambda(lambda x: x * 0.1)(out)

  output=merge([out,Input],mode='sum')
  output=BatchNormalization(axis=channel_axis)(output)
  output=Activation('relu')(output)

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
    output=BatchNormalization(axis=channel_axis)(output)
    return output


def Inception_B(Input):
    Input=Activation('relu')(Input)
    path1= Convolution2D(128//nb_filters_reduction_factor,1,1,border_mode='same',activation='relu')(Input)
    path1=Convolution2D(160//nb_filters_reduction_factor,1,7,border_mode='same',activation='relu')(path1)
    path1=Convolution2D(192//nb_filters_reduction_factor,7,1,border_mode='same',activation='relu')(path1)

    path2=Convolution2D(192//nb_filters_reduction_factor,1,1,border_mode='same',activation='relu')(Input)

    out=merge([path1,path2],mode='concat')
    out=Convolution2D(1152//nb_filters_reduction_factor,1,1,border_mode='same')(out)
    out = Lambda(lambda x: x * 0.1)(out)
    output=merge([out,Input],mode='sum')
    output=BatchNormalization(axis=channel_axis)(output)
    output=Activation('relu')(output)

    return output


def Reduction_B(Input):
    path1=Convolution2D(256//nb_filters_reduction_factor,1,1,border_mode='same',activation='relu')(Input)
    path1=Convolution2D(288//nb_filters_reduction_factor,3,3,border_mode='same',activation='relu')(path1)
    path1=Convolution2D(320//nb_filters_reduction_factor,3,3,subsample=(2,2),border_mode='valid',activation='relu')(path1)

    path2=Convolution2D(256//nb_filters_reduction_factor,1,1,border_mode='same',activation='relu')(Input)
    path2=Convolution2D(288//nb_filters_reduction_factor,3,3,subsample=(2,2),border_mode='valid',activation='relu')(path2)

    path3=Convolution2D(256//nb_filters_reduction_factor,1,1,border_mode='same',activation='relu')(Input)
    path3=Convolution2D(384//nb_filters_reduction_factor,3,3,subsample=(2,2),border_mode='valid',activation='relu')(path3)

    path4=MaxPooling2D((3,3),strides=(2,2))(Input)

    output=merge([path1,path2,path3,path4],mode='concat',concat_axis=-1)
    output=BatchNormalization(axis=channel_axis)(output)
    return output


def Inception_C(Input):
    Input=Activation('relu')(Input)
    path1=Convolution2D(192//nb_filters_reduction_factor,1,1,border_mode='same',activation='relu')(Input)
    path1=Convolution2D(224//nb_filters_reduction_factor,1,3,border_mode='same',activation='relu')(path1)
    path1=Convolution2D(256//nb_filters_reduction_factor,3,1,border_mode='same',activation='relu')(path1)

    path2=Convolution2D(192//nb_filters_reduction_factor,1,1,border_mode='same',activation='relu')(Input)
    out=merge([path1,path2],mode='concat',concat_axis=-1)
    out=Convolution2D(2144//nb_filters_reduction_factor,1,1,border_mode='same')(out)
    out = Lambda(lambda x: x * 0.1)(out)
    output=merge([out,Input],mode='sum')
    output=BatchNormalization(axis=channel_axis)(output)
    output=Activation('relu')(output)
    return output



img_rows, img_cols = 150, 150
img_channels = 3
k = 256
l = 256
m = 384
n = 384
# in original inception-v4, these are 4, 7, 3, respectively
num_A_blocks = 5
num_B_blocks = 10
num_C_blocks = 5
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
x=Flatten()(x)
predictions = Dense(nb_classes, activation='softmax')(x)
model = Model(input=inputs, output=predictions)
print(model.summary())
