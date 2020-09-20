import pandas as pd
import spacy
import torch.optim as optim
import torch.nn as nn
import random
from torchtext import data
import torch
import torchtext
from torchtext.datasets import text_classification
import os

# Set batch size
BATCH_SIZE = 8
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Reproducing same results
SEED = 2019

# Torch
torch.manual_seed(SEED)


# handling text data

TEXT = data.Field(batch_first=True, include_lengths=True,
                  sequential=True, tokenize='spacy', use_vocab=True)
LABEL = data.LabelField(dtype=torch.float, batch_first=True)

fields = [(None, None), ('label', LABEL), ('text', TEXT)]

# loading custom train dataset
training_data = data.TabularDataset(path=r'data/train_E6oV3lV.csv',
                                    format='csv',
                                    fields=fields,
                                    skip_header=True)

# print preprocessed text
print(vars(training_data.examples[0]))

train_data, valid_data = training_data.split(
    split_ratio=0.9, stratified=True, random_state=random.seed(SEED))

# check whether cuda is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Load an iterator
train_iterator, valid_iterator = data.BucketIterator.splits(
    (train_data, valid_data),
    batch_size=BATCH_SIZE,
    sort_key=lambda x: len(x.text),
    sort_within_batch=True,
    device=device)


# initialize glove embeddings
TEXT.build_vocab(train_data, min_freq=3, vectors="glove.6B.300d")
LABEL.build_vocab(train_data)

# No. of unique tokens in text
print("Size of TEXT vocabulary:", len(TEXT.vocab))

# No. of unique tokens in label
print("Size of LABEL vocabulary:", len(LABEL.vocab))

# Commonly used words
print(TEXT.vocab.freqs.most_common(10))

# Word dictionary
print(TEXT.vocab.stoi)

PAD_IDX = TEXT.vocab.stoi[TEXT.pad_token]


class classifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers,
                 bidirectional, dropout, pad_idx):

        super().__init__()

        self.embedding = nn.Embedding(
            vocab_size, embedding_dim, padding_idx=pad_idx)

        self.rnn = nn.LSTM(embedding_dim,
                           hidden_dim,
                           num_layers=n_layers,
                           bidirectional=bidirectional,
                           dropout=dropout)

        self.fc1 = nn.Linear(hidden_dim * 2, hidden_dim)

        self.fc2 = nn.Linear(hidden_dim, 1)

        self.dropout = nn.Dropout(dropout)

    def forward(self, text, text_lengths):

        # text = [sent len, batch size]

        embedded = self.embedding(text)

        # embedded = [sent len, batch size, emb dim]

        # pack sequence
        packed_embedded = nn.utils.rnn.pack_padded_sequence(
            embedded, text_lengths)

        packed_output, (hidden, cell) = self.rnn(packed_embedded)

        # unpack sequence
        # output, output_lengths = nn.utils.rnn.pad_packed_sequence(packed_output)

        # output = [sent len, batch size, hid dim * num directions]
        # output over padding tokens are zero tensors

        # hidden = [num layers * num directions, batch size, hid dim]
        # cell = [num layers * num directions, batch size, hid dim]

        # concat the final forward (hidden[-2,:,:]) and backward (hidden[-1,:,:]) hidden layers
        # and apply dropout

        hidden = self.dropout(
            torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1))
        output = self.fc1(hidden)
        output = self.dropout(self.fc2(output))

        # hidden = [batch size, hid dim * num directions]

        return output


# define hyperparameters
size_of_vocab = len(TEXT.vocab)
embedding_dim = 300
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
                   True,
                   dropout, PAD_IDX)

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
model.embedding.weight.data.copy_(pretrained_embeddings)

#  to initiaise padded to zeros
model.embedding.weight.data[PAD_IDX] = torch.zeros(embedding_dim)

print(model.embedding.weight.data)
model.to(device)  # CNN to GPU

# define optimizer and loss
optimizer = optim.Adam(model.parameters(), lr=0.001)
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
    print(
        f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc * 100:.2f}%')
    print(
        f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc * 100:.2f}%')


# inference
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


test = pd.read_csv(r'data/test_tweets_anuFYb8.csv')

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

submission = pd.DataFrame(data={'id': ids, 'label': labels})
submission.to_csv('sub.csv', index=False)
