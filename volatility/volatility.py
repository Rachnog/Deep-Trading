from utils import *
from feature_extractor import *

import matplotlib.pylab as plt

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.recurrent import LSTM, GRU
from keras.layers import Convolution1D, MaxPooling1D, AtrousConvolution1D, RepeatVector
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, CSVLogger
from keras.layers.wrappers import Bidirectional
from keras import regularizers
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import *
from keras.optimizers import RMSprop, Adam, SGD, Nadam
from keras.initializers import *
from keras.constraints import *

from keras import backend as K
import seaborn as sns
sns.despine()

def data2change(data):
    change = pd.DataFrame(data).pct_change()
    change = change.replace([np.inf, -np.inf], np.nan)
    change = change.fillna(0.).values.tolist()
    change = [c[0] for c in change]
    return change

WINDOW = 30
EMB_SIZE = 6
STEP = 1
FORECAST = 1

data_original = pd.read_csv('./data/GOOGL.csv')[::-1]

openp = data_original.ix[:, 'Open'].tolist()
highp = data_original.ix[:, 'High'].tolist()
lowp = data_original.ix[:, 'Low'].tolist()
closep = data_original.ix[:, 'Adj Close'].tolist()
volumep = data_original.ix[:, 'Volume'].tolist()

openp = data2change(openp)
highp = data2change(highp)
lowp = data2change(lowp)
closep = data2change(closep)
volumep = data2change(volumep)

volatility = []
for i in range(WINDOW, len(openp)):
    window = highp[i-WINDOW:i]
    volatility.append(np.std(window))

openp, highp, lowp, closep, volumep = openp[WINDOW:], highp[WINDOW:], lowp[WINDOW:], closep[WINDOW:], volumep[WINDOW:]

plt.plot(volatility)
plt.show()

X, Y = [], []
for i in range(0, len(data_original), STEP): 
    try:
        o = openp[i:i+WINDOW]
        h = highp[i:i+WINDOW]
        l = lowp[i:i+WINDOW]
        c = closep[i:i+WINDOW]
        v = volumep[i:i+WINDOW]

        volat = volatility[i:i+WINDOW]

        y_i = volatility[i+WINDOW+FORECAST]  
        x_i = np.column_stack((volat, o, h, l, c, v))

    except Exception as e:
        break

    X.append(x_i)
    Y.append(y_i)

X, Y = np.array(X), np.array(Y)
X_train, X_test, Y_train, Y_test = create_Xt_Yt(X, Y)

X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], EMB_SIZE))
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], EMB_SIZE))


epsilon = 1.0e-9
def qlike_loss(y_true, y_pred):
    y_pred = K.clip(y_pred, epsilon, 1.0 - epsilon)
    loss = K.log(y_pred) + y_true / y_pred
    return K.mean(loss, axis=-1)


def mse_log(y_true, y_pred):
    y_pred = K.clip(y_pred, epsilon, 1.0 - epsilon)
    loss = K.square(K.log(y_true) - K.log(y_pred))
    return K.mean(loss, axis=-1)


def mse_sd(y_true, y_pred):
    y_pred = K.clip(y_pred, epsilon, 1.0 - epsilon)
    loss = K.square(y_true - K.sqrt(y_pred))
    return K.mean(loss, axis=-1)    


def hmse(y_true, y_pred):
    y_pred = K.clip(y_pred, epsilon, 1.0 - epsilon)
    loss = K.square(y_true / y_pred - 1.)
    return K.mean(loss, axis=-1)    


def stock_loss(y_true, y_pred):
    alpha = 100.
    loss = K.switch(K.less(y_true * y_pred, 0), \
        alpha*y_pred**2 - K.sign(y_true)*y_pred + K.abs(y_true), \
        K.abs(y_true - y_pred)
        )
    return K.mean(loss, axis=-1)
    

model = Sequential()
model.add(Convolution1D(input_shape = (WINDOW, EMB_SIZE),
                        nb_filter=16,
                        filter_length=4,
                        border_mode='same'))
model.add(MaxPooling1D(2))
model.add(LeakyReLU())
model.add(Convolution1D(nb_filter=32,
                        filter_length=4,
                        border_mode='same'))
model.add(MaxPooling1D(2))
model.add(LeakyReLU())
model.add(Flatten())

model.add(Dense(16))
model.add(LeakyReLU())

model.add(Dense(1))
model.add(Activation('linear'))

opt = Nadam(lr=0.002)

reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.9, patience=25, min_lr=0.000001, verbose=1)
checkpointer = ModelCheckpoint(filepath="lolkekr.hdf5", verbose=1, save_best_only=True)

model.compile(optimizer=opt, 
              loss=mse_log)


try:
    history = model.fit(X_train, Y_train, 
              nb_epoch = 100, 
              batch_size = 256, 
              verbose=1, 
              validation_data=(X_test, Y_test),
              callbacks=[reduce_lr, checkpointer],
              shuffle=True)
except Exception as e:
    print e
finally:
    model.load_weights("lolkekr.hdf5")
    pred = model.predict(X_test)
    predicted = pred
    original = Y_test
    plt.title('Actual and predicted')
    plt.legend(loc='best')
    plt.plot(original, color='black', label = 'Original data')
    plt.plot(predicted, color='blue', label = 'Predicted data')
    plt.show()
    print np.mean(np.square(predicted - original))
    print np.mean(np.abs(predicted - original))
    print np.mean(np.abs((original - predicted) / original))