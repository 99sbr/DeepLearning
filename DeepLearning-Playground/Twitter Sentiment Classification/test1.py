
import thinc.extra.datasets

imdb_data = thinc.extra.datasets.imdb()
train_texts, train_labels = zip(*imdb_data[0])
print(train_texts[0])