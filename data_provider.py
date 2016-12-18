import json
import csv
import itertools
import numpy as np
import nltk
import sys
import os
from datetime import datetime
class Data_provider:

    def __init__(self):
        self._VOCABULARY_SIZE = int(os.environ.get('VOCABULARY_SIZE', '8000'))
        self._HIDDEN_DIM = int(os.environ.get('HIDDEN_DIM', '80'))
        self._LEARNING_RATE = float(os.environ.get('LEARNING_RATE', '0.005'))
        self._NEPOCH = int(os.environ.get('NEPOCH', '100'))
        self._MODEL_FILE = os.environ.get('MODEL_FILE')
        self.vocabulary_size = self._VOCABULARY_SIZE
        self.unknown_token = "UNKNOWN_TOKEN"
        self.sentence_start_token = "SENTENCE_START"
        self.sentence_end_token = "SENTENCE_END"

    def getdata(self, way = 1):
        if way == 2:
            with open('data.txt','r') as f:
                data = json.load(f)
                X_train = data['X']
                y_train = data['y']
            return X_train, y_train

        # Read the data and append SENTENCE_START and SENTENCE_END tokens
        print "Reading CSV file..."
        with open('reddit-comments-2015-08.csv', 'rb') as f:
            reader = csv.reader(f, skipinitialspace=True)
            reader.next()
            # Split full comments into sentences
            sentences = itertools.chain(*[nltk.sent_tokenize(x[0].decode('utf-8').lower()) for x in reader])
            # Append SENTENCE_START and SENTENCE_END
            sentences = ["%s %s %s" % (self.sentence_start_token, x, self.sentence_end_token) for x in sentences]
        print "Parsed %d sentences." % (len(sentences))

        # Tokenize the sentences into words
        tokenized_sentences = [nltk.word_tokenize(sent) for sent in sentences]

        # Count the word frequencies
        word_freq = nltk.FreqDist(itertools.chain(*tokenized_sentences))
        print "Found %d unique words tokens." % len(word_freq.items())

        # Get the most common words and build index_to_word and word_to_index vectors
        vocab = word_freq.most_common(self.vocabulary_size-1)
        index_to_word = [x[0] for x in vocab]
        index_to_word.append(self.unknown_token)
        word_to_index = dict([(w,i) for i,w in enumerate(index_to_word)])

        print "Using vocabulary size %d." % self.vocabulary_size
        print "The least frequent word in our vocabulary is '%s' and appeared %d times." % (vocab[-1][0], vocab[-1][1])

        # Replace all words not in our vocabulary with the unknown token
        for i, sent in enumerate(tokenized_sentences):
            tokenized_sentences[i] = [w if w in word_to_index else self.unknown_token for w in sent]

        # Create the training data
        X_train = np.asarray([[word_to_index[w] for w in sent[:-1]] for sent in tokenized_sentences])
        y_train = np.asarray([[word_to_index[w] for w in sent[1:]] for sent in tokenized_sentences])

        with open('data.txt', 'w') as outfile:
            json.dump({'X':X_train.tolist(), 'y':y_train.tolist()},outfile)
        return X_train, y_train
