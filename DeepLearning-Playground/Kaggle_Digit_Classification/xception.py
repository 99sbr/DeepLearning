from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten,Activation
from keras.layers.convolutional import Convolution2D ,SeparableConvolution2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.pooling  import GlobalMaxPooling2D
from keras.layers import Input , Merge
from keras.utils import np_utils
from keras import backend as K
from keras.layers.normalization import BatchNormalization

def Xception_Module():
	nb_filters=7

	Entry_Flow_Branch1=Sequential()
	Entry_Flow_Branch2=Sequential()
	Entry_Flow_Branch3=Sequential()
	Main_Flow_Path1=Sequential()
	Main_Flow_Path2=Sequential()
	Main_Flow_Path3=Sequential()
	Main_Flow_Path4=Sequential()
	Main_Flow_Path5=Sequential()
	Main_Flow_Path6=Sequential()

	Middle_Flow_Branch=Sequential()
	Exit_Flow_Branch=Sequential()

	Main_Flow_Path1.add(Convolution2D(nb_filters, 3, 3,border_mode='same',input_shape=(1, 28, 28)))
	Main_Flow_Path1.add(BatchNormalization())
	Main_Flow_Path1.add(Activation('relu'))

	Main_Flow_Path1.add(Convolution2D(nb_filters, 3, 3,border_mode='same'))
	Main_Flow_Path1.add(BatchNormalization())
	Main_Flow_Path1.add(Activation('relu'))

	Entry_Flow_Branch1.add(Main_Flow_Path1)
	Entry_Flow_Branch1.add(Convolution2D(nb_filters,1,1,subsample=(1,1)))
	Entry_Flow_Branch1.add(MaxPooling2D((2,2),strides=(1,1)))


	Main_Flow_Path1.add(SeparableConvolution2D(nb_filters, 3, 3,border_mode='same'))
	Main_Flow_Path1.add(BatchNormalization())
	Main_Flow_Path1.add(Activation('relu'))

	Main_Flow_Path1.add(SeparableConvolution2D(nb_filters,3,3,border_mode='same'))
	Main_Flow_Path1.add(MaxPooling2D((2,2),strides=(1,1)))
	Entry_Level_Branch1_merged = Merge([Main_Flow_Path1,Entry_Flow_Branch1], mode='sum')

	Main_Flow_Path2.add(Entry_Level_Branch1_merged)
	Main_Flow_Path2.add(BatchNormalization())
	Main_Flow_Path2.add(Activation('relu'))

	Main_Flow_Path2.add(SeparableConvolution2D(nb_filters,3,3,border_mode='same'))
	Main_Flow_Path2.add(BatchNormalization())
	Main_Flow_Path2.add(Activation('relu'))

	Main_Flow_Path2.add(SeparableConvolution2D(nb_filters,3,3,border_mode='same'))
	Main_Flow_Path2.add(MaxPooling2D((2,2),strides=(1,1)))

	Entry_Flow_Branch2.add(Entry_Level_Branch1_merged)
	Entry_Flow_Branch2.add(Convolution2D(nb_filters,1,1,subsample=(1,1),border_mode='same'))
	Entry_Flow_Branch2.add(MaxPooling2D((2,2),strides=(1,1)))
	Entry_Level_Branch2_merged=Merge([Main_Flow_Path2,Entry_Flow_Branch2],mode='sum')

	Main_Flow_Path3.add(Entry_Level_Branch2_merged)
	Main_Flow_Path3.add(BatchNormalization())
	Main_Flow_Path3.add(Activation('relu'))

	Main_Flow_Path3.add(SeparableConvolution2D(nb_filters,3,3,border_mode='same'))
	Main_Flow_Path3.add(BatchNormalization())
	Main_Flow_Path3.add(Activation('relu'))

	Main_Flow_Path3.add(SeparableConvolution2D(nb_filters,3,3,border_mode='same'))
	Main_Flow_Path3.add(MaxPooling2D((2,2),strides=(1,1)))

	Entry_Flow_Branch3.add(Entry_Level_Branch2_merged)
	Entry_Flow_Branch3.add(Convolution2D(nb_filters,1,1,subsample=(1,1),border_mode='same'))
	Entry_Flow_Branch3.add(MaxPooling2D((2,2),strides=(1,1)))
	Entry_Level_Branch3_merged=Merge([Main_Flow_Path3,Entry_Flow_Branch3],mode='sum')
	Main_Flow_Path4.add(Entry_Level_Branch3_merged)



	###### Middle Flow #########
	Middle_Flow_Branch.add(Main_Flow_Path4)
	Main_Flow_Path4.add(BatchNormalization())
	Main_Flow_Path4.add(Activation('relu'))
	Main_Flow_Path4.add(SeparableConvolution2D(nb_filters,3,3,border_mode='same'))

	Main_Flow_Path4.add(BatchNormalization())
	Main_Flow_Path4.add(Activation('relu'))
	Main_Flow_Path4.add(SeparableConvolution2D(nb_filters,3,3,border_mode='same'))

	Main_Flow_Path4.add(BatchNormalization())
	Main_Flow_Path4.add(Activation('relu'))
	Main_Flow_Path4.add(SeparableConvolution2D(nb_filters,3,3,border_mode='same'))

	Middle_Flow_Branch_merged=Merge([Main_Flow_Path4,Middle_Flow_Branch],mode='concat')

	Exit_Flow_Branch.add(Middle_Flow_Branch_merged)
	Exit_Flow_Branch.add(Convolution2D(nb_filters,1,1,subsample=(1,1)))
	Exit_Flow_Branch.add(MaxPooling2D((2,2),strides=(1,1)))
	Main_Flow_Path5.add(Middle_Flow_Branch_merged)

	Main_Flow_Path5.add(BatchNormalization())
	Main_Flow_Path5.add(Activation('relu'))
	Main_Flow_Path5.add(SeparableConvolution2D(nb_filters,3,3,border_mode='same'))
	Main_Flow_Path5.add(BatchNormalization())
	Main_Flow_Path5.add(Activation('relu'))
	Main_Flow_Path5.add(SeparableConvolution2D(nb_filters,3,3,border_mode='same'))
	Main_Flow_Path5.add(MaxPooling2D((2,2),strides=(1,1)))

	Exit_Flow_Branch_merged=Merge([Main_Flow_Path5,Exit_Flow_Branch],mode='concat')
	Main_Flow_Path6.add(Exit_Flow_Branch_merged)
	Main_Flow_Path6.add(SeparableConvolution2D(nb_filters,3,3,border_mode='same'))
	Main_Flow_Path6.add(BatchNormalization())
	Main_Flow_Path6.add(Activation('relu'))
	Main_Flow_Path6.add(SeparableConvolution2D(nb_filters,3,3,border_mode='same'))
	Main_Flow_Path6.add(BatchNormalization())
	Main_Flow_Path6.add(Activation('relu'))
	Main_Flow_Path6.add(GlobalMaxPooling2D())

	Main_Flow_Path6.add(Dropout(0.2))
	Main_Flow_Path6.add(Dense(112))
	Main_Flow_Path6.add(BatchNormalization())
	Main_Flow_Path6.add(Activation('relu'))
	Main_Flow_Path6.add(Dense(56))
	Main_Flow_Path6.add(BatchNormalization())
	Main_Flow_Path6.add(Activation('relu'))
	Main_Flow_Path6.add(Dense(10))
	Main_Flow_Path6.add(BatchNormalization())
	Main_Flow_Path6.add(Activation('softmax'))
	Main_Flow_Path6.compile(loss='categorical_crossentropy',optimizer='adam', metrics=['accuracy'])
	return Main_Flow_Path6