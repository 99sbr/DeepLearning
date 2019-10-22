from nltk.corpus import stopwords
import numpy as np
from random import shuffle
from tqdm import tqdm
np.random.seed(42)
import plac
import re
import nltk
from pathlib import Path
import thinc.extra.datasets
import pandas as pd
import spacy
from spacy.util import minibatch, compounding, decaying

n_jobs = 4
batch_size = 1000
dropout = decaying(0.6,0.2,1e-4)

def get_batches(train_data, model_type):
    max_batch_sizes = {'tagger': 32, 'parser': 16, 'ner': 16, 'textcat': 32}
    max_batch_size = max_batch_sizes[model_type]
    if len(train_data) < 1000:
        max_batch_size /= 2
    if len(train_data) < 500:
        max_batch_size /= 2
    batch_size = compounding(1, max_batch_size, 1.001)
    batches = minibatch(train_data, size=batch_size)
    return batches
# @plac.annotations(
#     model=("Model name. Defaults to blank 'en' model.", "option", "m", str),
#     output_dir=("Optional output directory", "option", "o", Path),
#     n_texts=("Number of texts to train from", "option", "t", int),
#     n_iter=("Number of training iterations", "option", "n", int))


def main(model='en_core_web_lg', output_dir='Model_Output', n_iter=10, n_texts=2000):
    if model is not None:
        nlp = spacy.load(model)  # load existing spaCy model
        print("Loaded model '%s'" % model)
    else:
        nlp = spacy.blank('en_core_web_lg')  # create blank Language class
        print("Created blank 'en' model")

    # add the text classifier to the pipeline if it doesn't exist
    # nlp.create_pipe works for built-ins that are registered with spaCy
    if 'textcat' not in nlp.pipe_names:
        textcat = nlp.create_pipe('textcat', )
        nlp.add_pipe(textcat, last=True)
    # otherwise, get it, so we can add labels to it
    else:
        textcat = nlp.get_pipe('textcat')

    # add label to text classifier

    textcat.add_label('NEGATIVE')


    (train_texts, train_cats), (dev_texts, dev_cats) = load_data(limit=n_texts)
    print("Using {} examples ({} training, {} evaluation)"
          .format(n_texts, len(train_texts), len(dev_texts)))
    train_data = list(zip(train_texts,
                          [{'cats': cats} for cats in train_cats]))

    # get names of other pipes to disable them during training
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe != 'textcat']
    with nlp.disable_pipes(*other_pipes):  # only train textcat
        optimizer = nlp.begin_training()
        #dropout = decaying(0.5, 0.2, 1e-4)

        print("Training the model...")
        print('{:^5}\t{:^5}\t{:^5}\t{:^5}'.format('LOSS', 'P', 'R', 'F'))
        for i in range(n_iter):

            losses = {}
            # batch up the examples using spaCy's minibatch
            batches = get_batches(train_data,'textcat')
            for batch in batches:
                texts, annotations = zip(*batch)
                nlp.update(texts, annotations, sgd=optimizer,
                           losses=losses, drop=0.2)
            with textcat.model.use_params(optimizer.averages):
                # evaluate on the dev data split off in load_data()
                scores = evaluate(nlp.tokenizer, textcat, dev_texts, dev_cats)
            print('{0:.3f}\t{1:.3f}\t{2:.3f}\t{3:.3f}'  # print a simple table
                  .format(losses['textcat'], scores['textcat_p'],
                          scores['textcat_r'], scores['textcat_f']))

    # test the trained model
    test = pd.read_csv('./data/new_test.csv')
    print("Text Preprocessing ====>")
    test = test.fillna("")
    stop = stopwords.words('english')
    test['tweet'] = test['tweet'].apply(
        lambda x: " ".join(x for x in x.split() if x not in stop))
    

    # test_text = "This movie sucked"

    if output_dir is not None:
        output_dir = Path(output_dir)
        if not output_dir.exists():
            output_dir.mkdir()
        with nlp.use_params(optimizer.averages):
            nlp.to_disk(output_dir)
        print("Saved model to", output_dir)

        print("############# \n \n \n ")

        # test the saved model
        test_text = "no comment!  in #australia   #opkillingbay #seashepherd #helpcovedolphins #thecove  #helpcovedolphins"
        print("Loading from", output_dir)
        nlp = spacy.load(output_dir)
        doc2 = nlp(test_text)
        print(test_text, doc2.cats)
        #####################################

        prediction = []
        for text in tqdm(test['tweet'].values):

            try:
                doc = nlp(text)
                if doc.cats['NEGATIVE'] > 0.4:
                    prediction.append(1)
                else:
                    prediction.append(0)
            except Exception as e:
                print(text)
                print(e)
                prediction.append(1)

        submission = pd.DataFrame()
        submission['id'] = test['id']
        submission['label'] = prediction
        submission.to_csv('submission_twitter_spacy5.csv', index=False)


def load_data(limit=0, split=0.8):
    train = pd.read_csv('./data/new_train.csv')
    train.sample(frac=1)
    labels = train.label
    cats = [{'NEGATIVE': bool(y)} for y in labels]
    train = train.drop('label', 1)
    data = train
    split = int(len(data) * split)

    # pre-processing
    print("Text Preprocessing ====>")
    
    stop = stopwords.words('english')
    data['tweet'] = data['tweet'].apply(lambda x: " ".join(x for x in x.split() if x not in stop))
    

    return (data['tweet'][:split], cats[:split]), (data['tweet'][split:], cats[split:])


def evaluate(tokenizer, textcat, texts, cats):
    docs = (tokenizer(text) for text in texts)
    tp = 1e-8  # True positives
    fp = 1e-8  # False positives
    fn = 1e-8  # False negatives
    tn = 1e-8  # True negatives
    for i, doc in enumerate(textcat.pipe(docs)):
        gold = cats[i]
        for label, score in doc.cats.items():
            if label not in gold:
                continue
            if score >= 0.5 and gold[label] >= 0.5:
                tp += 1.
            elif score >= 0.5 and gold[label] < 0.5:
                fp += 1.
            elif score < 0.5 and gold[label] < 0.5:
                tn += 1
            elif score < 0.5 and gold[label] >= 0.5:
                fn += 1
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f_score = 2 * (precision * recall) / (precision + recall)
    return {'textcat_p': precision, 'textcat_r': recall, 'textcat_f': f_score}


if __name__ == '__main__':
    main(model='en_core_web_lg', output_dir='Model_Output', n_iter=10, n_texts=1000)
