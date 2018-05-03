import sys
import warnings

if not sys.warnoptions:
    warnings.simplefilter("ignore")
from sklearn.feature_extraction.text import CountVectorizer
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM , Bidirectional ,GRU, TimeDistributed
from keras.layers import Flatten, Dropout , Activation , BatchNormalization
from keras.layers import Conv1D, GlobalMaxPooling1D , MaxPooling1D 
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
from keras.preprocessing.text import Tokenizer
import re
import pandas as pd
import numpy as np
from textblob import TextBlob
from nltk.corpus import stopwords
import numpy as np
from keras.callbacks import Callback
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score



MAX_NUM_WORDS = 10000
MAX_SEQUENCE_LENGTH=1000
EMBED_DIM = 100
LSTM_OUT = 196
BATCH_SIZE =64

train=pd.read_csv('train_E6oV3lV.csv')
test=pd.read_csv('test_tweets_anuFYb8.csv')

target=train.label
train=train.drop('label',1)
data=train.append(test)

# pre-processing
print("Text Preprocessing ====>")
data['tweet']=data['tweet'].apply(lambda x: x.lower())
data['tweet']=data['tweet'].apply((lambda x: re.sub('[^a-zA-z0-9\s]','',x)))
data['tweet']=data['tweet'].apply(lambda x: x.replace('user',''))
stop = stopwords.words('english')
data['tweet'] = data['tweet'].apply(lambda x: " ".join(x for x in x.split() if x not in stop))

# prepare tokenizer
t=Tokenizer(num_words=MAX_NUM_WORDS)
t.fit_on_texts(data['tweet'])
vocab_size=len(t.word_index)+1
print('VOCAB SIZE',vocab_size)
# integer encode the documents
encoded_docs = t.texts_to_sequences(data['tweet'])
padded_docs = pad_sequences(encoded_docs,  padding='post')
#padded_docs=np.hstack((padded_docs,sentiment))

# load the whole embedding into memory
embeddings_index = dict()
f = open('/Users/subir/Downloads/glove/glove.6B.100d.txt')
for line in f:
  values = line.split()
  word = values[0]
  coefs = np.asarray(values[1:], dtype='float32')
  embeddings_index[word] = coefs
f.close()
print('Loaded %s word vectors.' % len(embeddings_index))

# create a weight matrix for words in training docs
embedding_matrix = np.ones((vocab_size, EMBED_DIM))*0.01
for word, i in t.word_index.items():
  embedding_vector = embeddings_index.get(word)
  if embedding_vector is not None:
    embedding_matrix[i] = embedding_vector

train_Seq=padded_docs[:len(train)]
test_Seq=padded_docs[len(train):]

print("Model Training ---")

model = Sequential()
model.add(Embedding(vocab_size, EMBED_DIM, weights=[embedding_matrix] ,input_length = padded_docs.shape[1],trainable=False))

model.add(Bidirectional(GRU(LSTM_OUT,return_sequences=True)))
model.add(Dropout(0.2))
model.add(BatchNormalization())

model.add(Bidirectional(GRU(LSTM_OUT,return_sequences=True)))
model.add(Dropout(0.2))
model.add(BatchNormalization())

#model.add(Flatten())
# model.add(Conv1D(128,5,activation='relu'))
# model.add(MaxPooling1D(5))
# model.add(Conv1D(128,5,activation='relu'))
# model.add(MaxPooling1D(5))
# model.add(Conv1D(128,5,activation='relu'))
# model.add(GlobalMaxPooling1D())
model.add(Flatten())
model.add(Dense(2,activation='softmax'))
model.compile(loss = 'binary_crossentropy', optimizer='adam',metrics = ['accuracy'])


print(model.summary())

Y = pd.get_dummies(target).values
X_train, X_test, Y_train, Y_test = train_test_split(train_Seq,Y, test_size = 0.3, random_state = 42)


model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=5, batch_size=BATCH_SIZE)

prediction=model.predict_classes(test_Seq)
submission=pd.DataFrame()
submission['id']=test['id']
submission['label']=prediction
submission.to_csv('sub.csv',index=False)