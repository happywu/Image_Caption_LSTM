#! /usr/bin/env python

import csv
import itertools
import numpy as np
import nltk
import sys
import os
from datetime import datetime
import json
import cPickle as pickle
from LSTM import LSTM
from data_provider import Data_provider

data_provider = Data_provider()
X_train, y_train = data_provider.getdata(2)
X_train = np.array(X_train)
y_train = np.array(y_train)

np.random.seed(10)
# Train on a small subset of the data to see what happens
vocabulary_size = 8000
_HIDDEN_DIM = 80
LSTM_model = LSTM.init(vocabulary_size, _HIDDEN_DIM, vocabulary_size)
#print 'X_train', len(X_train[:100])
losses = LSTM.train_with_sgd(LSTM_model['model'], X_train[:100], y_train[:100], nepoch=2, evaluate_loss_after=1)
LSTM.grad_check(X_train[:100], y_train[:100], LSTM_model['model'])
#print losses
with open('model.pkl', 'w') as f:
    pickle.dump(LSTM_model['model'],f)

#with open('model.pkl', 'r') as f:
#    my_model = pickle.load(f)


#model = RNNNumpy(vocabulary_size, hidden_dim=_HIDDEN_DIM)
#t1 = time.time()
#model.sgd_step(X_train[10], y_train[10], _LEARNING_RATE)
#t2 = time.time()
#print "SGD Step time: %f milliseconds" % ((t2 - t1) * 1000.)

#if _MODEL_FILE != None:
   # load_model_parameters_theano(_MODEL_FILE, model)

#train_with_sgd(model, X_train, y_train, nepoch=_NEPOCH, learning_rate=_LEARNING_RATE)