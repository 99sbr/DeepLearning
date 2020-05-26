import torch
import torchtext
from torchtext.datasets import text_classification
import os
BATCH_SIZE = 8
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#Reproducing same results
SEED = 2019

#Torch
torch.manual_seed(SEED)

#Cuda algorithms
torch.backends.cudnn.deterministic = True

#handling text data
from torchtext import data

TEXT = data.Field(batch_first=True,include_lengths=True,sequential=True,tokenize='spacy',use_vocab=True)
LABEL = data.LabelField(dtype = torch.float,batch_first=True)

fields = [(None, None),('label', LABEL) ,('text',TEXT)]

# loading custom train dataset
training_data = data.TabularDataset(path='data/train_E6oV3lV.csv',
                                    format='csv',
                                    fields=fields,
                                    skip_header=True)

# print preprocessed text
print(vars(training_data.examples[0]))

import random
train_data, valid_data = training_data.split(split_ratio=0.9,stratified=True,random_state = random.seed(SEED))

# check whether cuda is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# set batch size
BATCH_SIZE = 8

# Load an iterator
train_iterator, valid_iterator = data.BucketIterator.splits(
    (train_data, valid_data),
    batch_size=BATCH_SIZE,
    sort_key=lambda x: len(x.text),
    sort_within_batch=True,
    device=device)


#initialize glove embeddings
TEXT.build_vocab(train_data,min_freq=3,vectors = "glove.twitter.27B.100d")
LABEL.build_vocab(train_data)

#No. of unique tokens in text
print("Size of TEXT vocabulary:",len(TEXT.vocab))

#No. of unique tokens in label
print("Size of LABEL vocabulary:",len(LABEL.vocab))

#Commonly used words
print(TEXT.vocab.freqs.most_common(10))

#Word dictionary
print(TEXT.vocab.stoi)

import torch.nn as nn


class classifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim,
                 n_layers, bidirectional, dropout):

        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim,
                            hidden_dim,
                            num_layers=n_layers,
                            bidirectional=bidirectional,
                            dropout=dropout,
                            batch_first=True)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        self.act = nn.Sigmoid()

    def forward(self, text, text_lengths):
        # text = [batch size,sent_length]
        embedded = self.embedding(text)
        # embedded = [batch size, sent_len, emb dim]
        # packed sequence
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded,
                                                            text_lengths,
                                                            batch_first=True)
        packed_output, (hidden, cell) = self.lstm(packed_embedded)
        # hidden = [batch size, num layers * num directions,hid dim]
        # cell = [batch size, num layers * num directions,hid dim]
        # concat the final forward and backward hidden state
        hidden = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)
        # hidden = [batch size, hid dim * num directions]
        dense_outputs = self.fc(hidden)
        # Final activation function
        outputs = self.act(dense_outputs)
        return outputs

# define hyperparameters
size_of_vocab = len(TEXT.vocab)
embedding_dim = 100
num_hidden_nodes = 512
num_output_nodes = 1
num_layers = 2
bidirection = True
dropout = 0.5

# instantiate the model
model = classifier(size_of_vocab,
                   embedding_dim,
                   num_hidden_nodes,
                   num_output_nodes,
                   num_layers,
                   bidirectional=True,
                   dropout=dropout)

# architecture
print(model)


# No. of trianable parameters
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


print(f'The model has {count_parameters(model):,} trainable parameters')

# Initialize the pretrained embedding
pretrained_embeddings = TEXT.vocab.vectors
model.embedding.weight.data.copy_(pretrained_embeddings)

print(pretrained_embeddings.shape)

import torch.optim as optim

# define optimizer and loss
optimizer = optim.Adam(model.parameters())
criterion = nn.BCELoss()
scheduler = optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.9)


# define metric
def binary_accuracy(preds, y):
    # round predictions to the closest integer
    rounded_preds = torch.round(preds)

    correct = (rounded_preds == y).float()
    acc = correct.sum() / len(correct)
    return acc


# push to cuda if available
model = model.to(device)
criterion = criterion.to(device)


def train(model, iterator, optimizer, criterion):
    # initialize every epoch
    epoch_loss = 0
    epoch_acc = 0

    # set the model in training phase
    model.train()

    for batch in iterator:
        # resets the gradients after every batch
        optimizer.zero_grad()

        # retrieve text and no. of words
        text, text_lengths = batch.text

        # convert to 1D tensor
        predictions = model(text, text_lengths).squeeze()

        # compute the loss
        loss = criterion(predictions, batch.label)

        # compute the binary accuracy
        acc = binary_accuracy(predictions, batch.label)

        # backpropage the loss and compute the gradients
        loss.backward()

        # update the weights
        optimizer.step()

        # loss and accuracy
        epoch_loss += loss.item()
        epoch_acc += acc.item()

    return epoch_loss / len(iterator), epoch_acc / len(iterator)

def evaluate(model, iterator, criterion):

    # initialize every epoch
    epoch_loss = 0
    epoch_acc = 0

    # deactivating dropout layers
    model.eval()

    # deactivates autograd
    with torch.no_grad():

        for batch in iterator:

            # retrieve text and no. of words
            text, text_lengths = batch.text

            # convert to 1d tensor
            predictions = model(text, text_lengths).squeeze()

            # compute loss and accuracy
            loss = criterion(predictions, batch.label)
            acc = binary_accuracy(predictions, batch.label)

            # keep track of loss and accuracy
            epoch_loss += loss.item()
            epoch_acc += acc.item()

    return epoch_loss / len(iterator), epoch_acc / len(iterator)


N_EPOCHS = 5
best_valid_loss = float('inf')

for epoch in range(N_EPOCHS):

    # train the model
    train_loss, train_acc = train(model, train_iterator, optimizer, criterion)

    # evaluate the model
    valid_loss, valid_acc = evaluate(model, valid_iterator, criterion)

    # save the best model
    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), 'saved_weights.pt')

    print('Epoch: {}\n'.format(epoch))
    print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc * 100:.2f}%')
    print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc * 100:.2f}%')


# inference
import spacy
nlp = spacy.load('en')


def predict(model, sentence):
    tokenized = [tok.text
                 for tok in nlp.tokenizer(sentence)]  # tokenize the sentence
    indexed = [TEXT.vocab.stoi[t]
               for t in tokenized]  # convert to integer sequence
    length = [len(indexed)]  # compute no. of words
    tensor = torch.LongTensor(indexed).to(device)  # convert to tensor
    tensor = tensor.unsqueeze(1).T  # reshape in form of batch,no. of words
    length_tensor = torch.LongTensor(length)  # convert to tensor
    prediction = model(tensor, length_tensor)  # prediction
    return prediction.item()

import pandas as pd
test = pd.read_csv('data/test_tweets_anuFYb8.csv')

prediction = []
ids = []
for row in (test.iterrows()):
    ids.append(row[1]['id'])
    prediction.append(predict(model, row[1]['tweet']))

labels = []
for pred in prediction:
    if pred > 0.5:
        labels.append(1)
    else:
        labels.append(0)

submission = pd.DataFrame(data={'id':ids,'label':labels})
submission.to_csv('sub.csv',index=False)