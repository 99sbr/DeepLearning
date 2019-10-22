import sys
import warnings

if not sys.warnoptions:
    warnings.simplefilter("ignore")
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Embedding, Bidirectional, GRU, LSTM
from keras.layers import Flatten, Dropout, BatchNormalization
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
import re
import pandas as pd
import numpy as np
import spacy
np.random.seed(42)
from keras.optimizers import Adam
from nltk.corpus import stopwords
from keras.layers import TimeDistributed


def get_embeddings(vocab):
    # max_rank = max(lex.rank for lex in vocab if lex.has_vector)
    # vectors = np.ndarray((max_rank+1, vocab.vectors_length), dtype='float32')
    # for lex in vocab:
    #     if lex.has_vector:
    #         vectors[lex.rank] = lex.vector
    return vocab.vectors.data

def get_features(docs, max_length):
    docs = list(docs)
    Xs = np.zeros((len(docs), max_length), dtype='int32')
    for i, doc in enumerate(docs):
        j = 0
        for token in doc:
            vector_id = token.vocab.vectors.find(key=token.orth)
            if vector_id >= 0:
                Xs[i, j] = vector_id
            else:
                Xs[i, j] = 0
            j += 1
            if j >= max_length:
                break
    return Xs

def get_labelled_sentences(docs, doc_labels):
    labels = []
    sentences = []
    for doc, y in zip(docs, doc_labels):
        for sent in doc.sents:
            sentences.append(sent)
            labels.append(y)
    return sentences, np.asarray(labels, dtype='int32')

MAX_NUM_WORDS = 100
MAX_SEQUENCE_LENGTH = 100
EMBED_DIM = 25
LSTM_OUT = 256
BATCH_SIZE = 32

train = pd.read_csv('train_E6oV3lV.csv')
test = pd.read_csv('test_tweets_anuFYb8.csv')

target = train.label
train = train.drop('label', 1)
data = train.append(test)

# pre-processing

print("Text Preprocessing ====>")
data['tweet'] = data['tweet'].str.replace('[^\w\s]', '')
data['tweet'] = data['tweet'].apply(lambda x: x.lower())
data['tweet'] = data['tweet'].apply(
    (lambda x: re.sub('[^a-zA-z0-9\s]', '', x)))
data['tweet'] = data['tweet'].apply(lambda x: x.replace('user', ''))
stop = stopwords.words('english')
data['tweet'] = data['tweet'].apply(lambda x: " ".join(x for x in x.split() if x not in stop))


# from sklearn.preprocessing import LabelEncoder
# labelencoder_y_1 = LabelEncoder()
# target = labelencoder_y_1.fit_transform(target)
# prepare tokenizer
# t = Tokenizer(num_words=MAX_NUM_WORDS)
# t.fit_on_texts(data['tweet'])
# vocab_size = len(t.word_index) + 1
# print('VOCAB SIZE', vocab_size)
# # integer encode the documents
# encoded_docs = t.texts_to_sequences(data['tweet'])
# padded_docs = pad_sequences(encoded_docs, padding='post')
# padded_docs=np.hstack((padded_docs,sentiment))

# load the whole embedding into memory
# embeddings_index = dict()
# f = open('/Users/subir/WordEmbeddings/glove/glove/glove.twitter.27B.25d.txt')
# for line in f:
#     values = line.split()
#     word = values[0]
#     coefs = np.asarray(values[1:], dtype='float32')
#     embeddings_index[word] = coefs
# f.close()


print("Loading spaCy")
nlp = spacy.load('en_vectors_web_lg', parser=False, tagger=False, entity=False)
nlp.add_pipe(nlp.create_pipe('sentencizer'))
embeddings = get_embeddings(nlp.vocab)
print('Loaded %s word vectors.' % len(embeddings))
print(embeddings.shape[0],embeddings.shape[1])

# create a weight matrix for words in training docs
# embedding_matrix = np.ones((vocab_size, EMBED_DIM)) * 0.01
# for word, i in t.word_index.items():
#     embedding_vector = embeddings_index.get(word)
#     if embedding_vector is not None:
#         embedding_matrix[i] = embedding_vector

train_texts = data[:len(train)]
test_text = data[len(train):]

train_texts = train_texts['tweet'].values
test_text = test_text['tweet'].values


print("Parsing texts...")
target =  np.asarray(target, dtype='int32')
X_train, X_dev, Y_train, Y_dev = train_test_split(train_texts, target, test_size=0.3, random_state=42)

train_docs = list(nlp.pipe(X_train))
dev_docs = list(nlp.pipe(X_dev))
test_docs = list(nlp.pipe(test_text))



if True:
    train_docs, train_labels = get_labelled_sentences(train_docs, Y_train)
    dev_docs, dev_labels = get_labelled_sentences(dev_docs, Y_dev)

train_X = get_features(train_docs, MAX_SEQUENCE_LENGTH)
dev_X = get_features(dev_docs, MAX_SEQUENCE_LENGTH)
test_X = get_features(test_docs,MAX_SEQUENCE_LENGTH)



print("Model Training ---")
model = Sequential()
model.add(
    Embedding(embeddings.shape[0], embeddings.shape[1], weights=[embeddings], input_length=MAX_SEQUENCE_LENGTH, trainable=False))
TimeDistributed(Dense(LSTM_OUT, use_bias=False))

model.add(Bidirectional(LSTM(LSTM_OUT,recurrent_dropout=0.4, dropout=0.4)))

model.add(Dense(32, activation='relu'))
model.add(Dense(1,activation='softmax'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())


model.fit(train_X, train_labels, validation_data=(dev_X, dev_labels), epochs=2, batch_size=BATCH_SIZE)

prediction = model.predict_classes(test_X)
submission = pd.DataFrame()
submission['id'] = test['id']
submission['label'] = prediction
submission.to_csv('submission_twitter_spacy+keras.csv', index=False)
