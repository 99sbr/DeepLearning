from keras import applications
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Model
from keras.layers import Dropout, Flatten, Dense

base_model = applications.InceptionV3(weights="imagenet", include_top=False, input_shape=(250, 250, 3))
x = base_model.output
x = Flatten()(x)
x = Dense(1024, activation='relu')(x)
x = Dropout(0.5)(x)
x = Dense(1024, activation='relu')(x)
# using sigmoid for multi-label classification
predictions = Dense(10, activation='sigmoid')(x)
# Creating final model
model = Model(inputs=base_model.input, outputs=predictions)

for layer in model.layers[:-5]:
    layer.trainable = False


model.compile(loss='binary_crossentropy', optimizer=optimizers.Adam(), metrics=['accuracy'])


print(model.summary())