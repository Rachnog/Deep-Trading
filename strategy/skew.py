import numpy as np
import json
import glob

import pandas as pd
import matplotlib.pylab as plt

from keras.models import Sequential
from keras.models import Model
from keras.layers.core import Dense, Dropout, Activation, Flatten, Permute, Reshape, Lambda
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
from keras.layers.noise import *


from keras import backend as K
import seaborn as sns
sns.despine()



from pyti.williams_percent_r import williams_percent_r
from pyti.relative_strength_index import relative_strength_index

import nolds


def data2change(data):
    change = pd.DataFrame(data).pct_change()
    change = change.replace([np.inf, -np.inf], np.nan)
    change = change.fillna(0.).values.tolist()
    change = [c[0] for c in change]
    return change


def remap(x, in_min, in_max, out_min, out_max):
  return (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min



def moving_average_convergence(group, nslow=12, nfast=6):
    emaslow = pd.ewma(group, span=nslow, min_periods=1).values.tolist()
    emafast = pd.ewma(group, span=nfast, min_periods=1).values.tolist()
    return np.array(emafast) -np.array(emaslow)


def shuffle_in_unison(a, b):
    # courtsey http://stackoverflow.com/users/190280/josh-bleecher-snyder
    assert len(a) == len(b)
    shuffled_a = np.empty(a.shape, dtype=a.dtype)
    shuffled_b = np.empty(b.shape, dtype=b.dtype)
    permutation = np.random.permutation(len(a))
    for old_index, new_index in enumerate(permutation):
        shuffled_a[new_index] = a[old_index]
        shuffled_b[new_index] = b[old_index]
    return shuffled_a, shuffled_b
 
 
def create_Xt_Yt(X, y, percentage=0.95):
    p = int(len(X) * percentage)
    X_train = X[0:p]
    Y_train = y[0:p]
     
    X_train, Y_train = shuffle_in_unison(X_train, Y_train)
 
    X_test = X[p:]
    Y_test = y[p:]

    return X_train, X_test, Y_train, Y_test


TEST_SET = 0.9
WINDOW = 30
STEP = 1
FORECAST = 14
ROLLING = 30

data_original = pd.read_csv('./data/AAPL1216.csv')[::-1]

openp = data_original.ix[:, 'Open'].tolist()
highp = data_original.ix[:, 'High'].tolist()
lowp = data_original.ix[:, 'Low'].tolist()
closep = data_original.ix[:, 'Adj Close'].tolist()
volumep = data_original.ix[:, 'Volume'].tolist()


ma30 = pd.DataFrame(closep).rolling(14).mean().values.tolist()
ma30 = [v[0] for v in ma30]
ma60 = pd.DataFrame(closep).rolling(30).mean().values.tolist()
ma60 = [v[0] for v in ma60]

nine_period_high = pd.rolling_max(pd.DataFrame(highp), window= ROLLING / 2)
nine_period_low = pd.rolling_min(pd.DataFrame(lowp), window=  ROLLING / 2)
ichimoku = (nine_period_high + nine_period_low) /2
ichimoku = ichimoku.replace([np.inf, -np.inf], np.nan)
ichimoku = ichimoku.fillna(0.).values.tolist()

macd_indie = moving_average_convergence(pd.DataFrame(closep))

wpr = williams_percent_r(closep)
rsi = relative_strength_index(closep,  ROLLING / 2)

volatility1 = pd.DataFrame(closep).rolling(ROLLING).std().values#.tolist()
volatility2 = pd.DataFrame(closep).rolling(ROLLING).var().values#.tolist()

volatility = volatility1 / volatility2
volatility = [v[0] for v in volatility]

rolling_skewness = pd.DataFrame(closep).rolling(ROLLING).skew().values 
rolling_kurtosis = pd.DataFrame(closep).rolling(ROLLING).kurt().values 


X, Y = [], []
for i in range(WINDOW, len(data_original)-1, STEP): 
    try:
        o = openp[i:i+WINDOW]
        h = highp[i:i+WINDOW]
        l = lowp[i:i+WINDOW]
        c = closep[i:i+WINDOW]
        v = volumep[i:i+WINDOW]
        volat = volatility[i:i+WINDOW]
        rsk = rolling_skewness[i:i+WINDOW]
        rku = rolling_kurtosis[i:i+WINDOW]
        macd = macd_indie[i:i+WINDOW]
        williams = wpr[i:i+WINDOW]
        relative = rsi[i:i+WINDOW]
        ichi = ichimoku[i:i+WINDOW]

        if closep[i+WINDOW+FORECAST] < 0.0001 or closep[i+WINDOW+FORECAST] > 10000:
          continue


        macd = remap(np.array(macd), np.array(macd).min(), np.array(macd).max(), -1, 1)
        williams = remap(np.array(williams), np.array(williams).min(), np.array(williams).max(), -1, 1)
        relative = remap(np.array(relative), np.array(relative).min(), np.array(relative).max(), -1, 1)
        ichi = remap(np.array(ichi), np.array(ichi).min(), np.array(ichi).max(), -1, 1)
        o = remap(np.array(o), np.array(o).min(), np.array(o).max(), -1, 1)
        h = remap(np.array(h), np.array(h).min(), np.array(h).max(), -1, 1)
        l = remap(np.array(l), np.array(l).min(), np.array(l).max(), -1, 1)
        c = remap(np.array(c), np.array(c).min(), np.array(c).max(), -1, 1)
        v = remap(np.array(v), np.array(v).min(), np.array(v).max(), -1, 1)
        volat = remap(np.array(volat), np.array(volat).min(), np.array(volat).max(), -1, 1)
        rsk = remap(np.array(rsk), np.array(rsk).min(), np.array(rsk).max(), -1, 1)
        rku = remap(np.array(rku), np.array(rku).min(), np.array(rku).max(), -1, 1)

        x_i = np.column_stack((o, h, l, c, v, volat, rsk, rku, macd, williams, relative, ichi))
        # x_i = np.column_stack((o, h, l, c, v))
        x_i = x_i.flatten()

        # y_i = (closep[i+WINDOW+FORECAST] - closep[i+WINDOW]) / closep[i+WINDOW]
        y_i = rolling_skewness[i+WINDOW+FORECAST]

        # y_i = nolds.hurst_rs(closep[i:i+WINDOW+FORECAST])


        if np.isnan(x_i).any() or np.isinf(x_i).any():
          continue


        if np.isnan(y_i).any() or np.isinf(y_i).any():
            continue                

    except Exception as e:
        print e
        break

    X.append(x_i)
    Y.append(y_i)


X, Y = np.array(X), np.array(Y)
X_train, X_test, Y_train, Y_test = create_Xt_Yt(X, Y, TEST_SET)


main_input = Input(shape=(len(X[0]), ), name='main_input')
x = GaussianNoise(0.05)(main_input)
x = Lambda(lambda x: K.clip(x, min_value=-1, max_value=1))(x)
x = Dense(64, activation='relu')(x)
x = GaussianNoise(0.05)(x)
output = Dense(1, activation = "linear", name = "out")(x)

final_model = Model(inputs=[main_input], outputs=[output])

opt = Adam(lr=0.002)

reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.9, patience=10, min_lr=0.000001, verbose=1)
checkpointer = ModelCheckpoint(monitor='val_loss', filepath="xxx.hdf5", verbose=1, save_best_only=True)

final_model.compile(optimizer=opt, 
              loss='mse')


for layer in final_model.layers:
    print layer, layer.output_shape

try:
    history = final_model.fit(X_train, Y_train, 
              nb_epoch = 500, 
              batch_size = 256, 
              verbose=1, 
              validation_data=(X_test, Y_test),
              callbacks=[reduce_lr, checkpointer],
              shuffle=True)

except Exception as e:
    print e
finally:
    final_model.load_weights("xxx.hdf5")
    pred = final_model.predict(X_test)



predicted = pred
original = Y_test

original_to_show = np.column_stack((closep, ma30, ma60))[WINDOW:]
original_to_show = original_to_show[int(len(X) * TEST_SET):] - 80
intersections = [0.]
for i, (a, b) in enumerate(zip(original_to_show[:, 1], original_to_show[:, 2])):
    if abs(a - b) < .5 and abs(i - intersections[-1]) > 5:
        intersections.append(i)

diff = abs(len(original_to_show) - len(predicted))
zero_first = np.zeros((diff - FORECAST))

original = remap(np.array(original), np.array(original).min(), np.array(original).max(), 0, 10)
predicted = remap(np.array(predicted), np.array(predicted).min(), np.array(predicted).max(), 0, 10)

original = np.append(zero_first, original)
predicted = np.append(zero_first, predicted)

plt.title('Actual and predicted')
plt.legend(loc='best')
plt.plot(original, color='black', label = 'Original data')
plt.plot(original_to_show)

indicator = original

colors = ['green', 'orange', 'red']
styles = ['-', '--']
for i in intersections:
    try:
        if indicator[i] < 4:
            color = colors[0]
            style = styles[1]
        elif indicator[i] >= 4 and indicator[i] < 7:
            color = colors[1]
            style = styles[0]
        else:
            color = colors[2]
            style = styles[0]
        plt.axvline(i, color = color, linestyle = style)
    except:
        continue


plt.show()   

print np.mean(np.square(pred - Y_test))
print np.mean(np.abs(pred - Y_test))
print np.mean(np.abs((Y_test - pred) / Y_test))
