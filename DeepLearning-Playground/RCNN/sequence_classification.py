import numpy as np 
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM , Bidirectional
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence

# seed
np.random.seed(1234)

top_words=10000
(X_train,y_train),(X_test,y_test)= imdb.load_data(num_words=top_words)

#truncate and pad input sequences
max_review_length = 200
X_train =sequence.pad_sequences(X_train,maxlen=max_review_length)
X_test  =sequence.pad_sequences(X_test,maxlen=max_review_length)

# create the model
embedding_vector_length = 32
model = Sequential()
model.add(Embedding(top_words, embedding_vector_length, input_length=max_review_length))
model.add(Bidirectional(LSTM(100)))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=5, batch_size=32)

# Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=2)
print("Accuracy Bidirectional: %.2f%%" % (scores[1]*100))