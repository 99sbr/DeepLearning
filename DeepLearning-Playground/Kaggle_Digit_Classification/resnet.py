from keras.models import Sequential
from keras.layers import Dense,Dropout,ZeroPadding2D,Flatten,Activation,Input , merge
from keras.layers.convolutional import Convolution2D ,SeparableConvolution2D,MaxPooling2D
from keras.layers.pooling  import GlobalMaxPooling2D
from keras.utils import np_utils
from keras import backend as K
from keras.layers.normalization import BatchNormalization

nb_filters=7
nb_conv=3


def ResNet(input_img):
    layer1=Convolution2D(nb_filters,nb_conv,nb_conv,subsample=(1,1),activation='relu')(input_img)  #26
    layer2=MaxPooling2D((2, 2), strides=(1, 1))(layer1) #25
    layer3=Convolution2D(nb_filters,nb_conv,nb_conv,activation='relu')(layer2) #23
    layer4=Convolution2D(nb_filters,nb_conv,nb_conv,activation='relu')(layer3) #21
    layer5=(MaxPooling2D((2,2),strides=(1,1)))(layer4)# 20

    layer6=Convolution2D(nb_filters,5,5,activation='relu')(layer1)
    layer7=MaxPooling2D((3,3),strides=(1,1))(layer6)
    layer8 = merge([layer5,layer7],mode='sum')

    layer9=Convolution2D(nb_filters,nb_conv,nb_conv,activation='relu')(layer8)
    layer10=Convolution2D(nb_filters,nb_conv,nb_conv,activation='relu')(layer9)
    layer11=MaxPooling2D((2,2),strides=(1,1))(layer10)

    layer12=Convolution2D(nb_filters,5,5,activation='relu')(layer8)
    layer13=MaxPooling2D((2,2),strides=(1,1))(layer12)

    layer14=merge([layer13,layer11])


    return layer8