import numpy as np
import json
import glob

from utils import *

import pandas as pd
import matplotlib.pylab as plt

from keras.models import Sequential
from keras.models import Model
from keras.layers.core import Dense, Dropout, Activation, Flatten, Permute, Reshape
from keras.layers import Merge, Input, concatenate
from keras.layers.recurrent import LSTM, GRU, SimpleRNN
from keras.layers import Convolution1D, MaxPooling1D, GlobalAveragePooling1D, GlobalMaxPooling1D, RepeatVector, AveragePooling1D
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, CSVLogger
from keras.layers.wrappers import Bidirectional, TimeDistributed
from keras import regularizers
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import *
from keras.optimizers import RMSprop, Adam, SGD, Nadam
from keras.initializers import *
from keras.constraints import *
from keras import regularizers
from keras import losses

from keras import backend as K
import seaborn as sns
sns.despine()

from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from sklearn.metrics import roc_auc_score
import sys


STEP = 1
FORECAST = 10

data_original = pd.read_csv('./bitcoin-historical-data/krakenUSD_1-min_data_2014-01-07_to_2017-05-31.csv')[-10000:]
data_original = data_original.dropna()


def prepare_data(window):

    openp = data_original.ix[:, 'Open'].tolist()[1:]
    highp = data_original.ix[:, 'High'].tolist()[1:]
    lowp = data_original.ix[:, 'Low'].tolist()[1:]
    closep = data_original.ix[:, 'Close'].tolist()[1:]
    volumep = data_original.ix[:, 'Volume_(BTC)'].tolist()[1:]
    volumecp = data_original.ix[:, 'Volume_(Currency)'].tolist()[1:]

    volatility = pd.DataFrame(closep).rolling(window).std().values.tolist()
    volatility = [v[0] for v in volatility]

    X, Y = [], []
    for i in range(0, len(data_original), STEP): 
        try:
            o = openp[i:i+window]
            h = highp[i:i+window]
            l = lowp[i:i+window]
            c = closep[i:i+window]
            v = volumep[i:i+window]
            vc = volumecp[i:i+window]
            volat = volatility[i:i+window]

            o = (np.array(o) - np.mean(o)) / np.std(o)
            h = (np.array(h) - np.mean(h)) / np.std(h)
            l = (np.array(l) - np.mean(l)) / np.std(l)
            c = (np.array(c) - np.mean(c)) / np.std(c)
            v = (np.array(v) - np.mean(v)) / np.std(v)
            vc = (np.array(vc) - np.mean(vc)) / np.std(vc)
            volat = (np.array(volat) - np.mean(volat)) / np.std(volat)

            x_i = np.column_stack((o, h, l, c, v, vc, volat))
            x_i = x_i.flatten()

            y_i = (closep[i+window+FORECAST] - closep[i+window]) / closep[i+window]

            if np.isnan(x_i).any():
                continue

        except Exception as e:
            break

        X.append(x_i)
        Y.append(y_i)

    X, Y = np.array(X), np.array(Y)
    X_train, X_test, Y_train, Y_test = create_Xt_Yt(X, Y)
    return X_train, X_test, Y_train, Y_test


def experiment(params):
    X_train, X_test, Y_train, Y_test = prepare_data(params['window'])

    print 'Trying', params


    try:
        main_input = Input(shape=(len(X_train[0]), ), name='main_input')
        x = Dense(params['units1'], activation=params['activation'])(main_input)
        x = Dense(params['units2'], activation=params['activation'])(x)
        x = Dense(params['units3'], activation=params['activation'])(x)

        output = Dense(1, activation = "linear", name = "out")(x)
        final_model = Model(inputs=[main_input], outputs=[output])
        opt = Adam(lr=params['lr'])

        final_model.compile(optimizer=opt,  loss=params['loss'])

        history = final_model.fit(X_train, Y_train, 
                  epochs = 5, 
                  batch_size = 256, 
                  verbose=1, 
                  validation_data=(X_test, Y_test),
                  shuffle=True)

        pred = final_model.predict(X_test)

    except:
        print 'Something happened'
        print '-' * 10

        return {'loss': 999999, 'status': STATUS_OK}


    predicted = pred
    original = Y_test

    mse = np.mean(np.square(predicted - original))    

    if np.isnan(mse):
        print 'NaN happened'
        print '-' * 10
        return {'loss': 999999, 'status': STATUS_OK}

    print mse
    print '-' * 10

    sys.stdout.flush() 
    return {'loss': mse, 'status': STATUS_OK}



space = {'window': hp.choice('window',[30, 60, 120, 180]),
        'units1': hp.choice('units1', [64, 512]),
        'units2': hp.choice('units2', [64, 512]),
        'units3': hp.choice('units3', [64, 512]),

        'lr': hp.choice('lr',[0.01, 0.001, 0.0001]),
        'activation': hp.choice('activation',['relu',
                                                'sigmoid',
                                                'tanh',
                                                'linear']),
        'loss': hp.choice('loss', [losses.logcosh,
                                    losses.mse,
                                    losses.mae,
                                    losses.mape])
        }



trials = Trials()
best = fmin(experiment, space, algo=tpe.suggest, max_evals=50, trials=trials)
print 'best: '
print best
