from process_data import *
from utils import *

from keras.models import Model
from keras.layers import Input
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution1D, MaxPooling1D, AtrousConvolution1D, RepeatVector
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, CSVLogger
from keras.layers.normalization import BatchNormalization

from keras.layers.advanced_activations import *
from keras.optimizers import Nadam
from keras.initializers import *

#Visualization
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split


def data2change(data):
    change = pd.DataFrame(data).pct_change()
    change = change.replace([np.inf, -np.inf], np.nan)
    change = change.fillna(0.).values.tolist()
    change = [c[0] for c in change]
    return change


WINDOW = 30
EMB_SIZE = 5
STEP = 1
FORECAST = 1

data_original = pd.read_csv('./DJIA_table.csv')[::-1]

openp = data_original.ix[:, 'Open'].tolist()
highp = data_original.ix[:, 'High'].tolist()
lowp = data_original.ix[:, 'Low'].tolist()
closep = data_original.ix[:, 'Adj Close'].tolist()
volumep = data_original.ix[:, 'Volume'].tolist()

highp = data2change(highp)
lowp = data2change(lowp)
openp = data2change(openp)
closep = data2change(closep)
volumep = data2change(volumep)


X, Y = [], []
for i in range(0, len(data_original), STEP): 
    try:
        o = openp[i:i+WINDOW]
        h = highp[i:i+WINDOW]
        l = lowp[i:i+WINDOW]
        c = closep[i:i+WINDOW]
        v = volumep[i:i+WINDOW]

        o = (np.array(o) - np.mean(o)) / np.std(o)
        h = (np.array(h) - np.mean(h)) / np.std(h)
        l = (np.array(l) - np.mean(l)) / np.std(l)
        c = (np.array(c) - np.mean(c)) / np.std(c)
        v = (np.array(v) - np.mean(v)) / np.std(v)

        y_i1 = closep[i+WINDOW+FORECAST]  

        if last_close < next_close:
            y_i2 = [1, 0]
        else:
            y_i2 = [0, 1] 

        x_i = np.column_stack((o, h, l, c, v))

    except Exception as e:
        break

    X.append(x_i)
    Y.append([y_i1, y_i2])

train_test_split = 0.8

X_train, X_test = X[:int(train_test_split * len(X))], X[int(train_test_split * len(X)):]
Y_train, Y_test = Y[:int(train_test_split * len(X))], Y[int(train_test_split * len(X)):]

Y_train1, Y_train2 = [y[0] for y in Y_train], [y[1] for y in Y_train]
Y_test1, Y_test2 = [y[0] for y in Y_test], [y[1] for y in Y_test]

X_train, X_test = np.array(X_train), np.array(X_test)
Y_train1, Y_train2 = np.array(Y_train1), np.array(Y_train2)
Y_test1, Y_test2 = np.array(Y_test1), np.array(Y_test2)

X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], EMB_SIZE))
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], EMB_SIZE))




main_input = Input(shape=(WINDOW, EMB_SIZE), name='ts_input')

lstm1 = LSTM(32, return_sequences=False, recurrent_dropout=0.5, dropout=0.5,
                    bias_initializer='ones')(main_input)

x = Dense(64)(lstm1)
x = PReLU()(x)
x = Dense(1, activation = 'linear', name = 'regression_out')(x)

final_model = Model(inputs=[main_input], 
              outputs=[x])


opt = Nadam(lr=0.002)

reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.9, patience=50, min_lr=0.000001, verbose=1)
checkpointer = ModelCheckpoint(monitor='val_loss', filepath="test_Multi24.hdf5", verbose=1, save_best_only=True)

final_model.compile(optimizer=opt, 
              loss='mse')

for layer in final_model.layers:
    print layer, layer.output_shape


try:
    history = final_model.fit(X_train, Y_train, 
              nb_epoch = 1000, 
              batch_size = 256, 
              verbose=1, 
              validation_data=([X_test2, X_test1], Y_test),
              callbacks=[reduce_lr, checkpointer],
              shuffle=True)

except Exception as e:
    print e

finally:
    final_model.load_weights("test_Multi24.hdf5")
    pred = final_model.predict([X_test2, X_test1])

    predicted = pred
    original = Y_test


    plt.title('Actual and predicted')
    plt.legend(loc='best')
    plt.plot(original, color='black', label = 'Original data')
    plt.plot(pred, color='blue', label = 'Predicted data')
    plt.show()


    print np.mean(np.square(predicted - original))
    print np.mean(np.abs(predicted - original))
    print np.mean(np.abs((original - predicted) / original))








# conv1 = Convolution1D(nb_filter=16,
#                       filter_length=4,
#                       border_mode='same')(main_input)
# activ1 = LeakyReLU()(conv1)
# conv_drop = Dropout(0.9)(activ1)
# conv2 = Convolution1D(nb_filter=8,
#                       filter_length=4,
#                       border_mode='same')(conv_drop)
# activ2 = LeakyReLU()(conv2)
# flat = Flatten()(activ2)
# conv_drop_out = Dropout(0.9)(flat)

# mlp1 = Dense(64)(conv_drop_out)
# mlp1 = LeakyReLU()(mlp1)
# mlp1 = Dense(1, activation = 'linear', name = 'regression_out')(mlp1)
    
# mlp2 = Dense(64)(conv_drop_out)
# mlp2 = BatchNormalization()(mlp2)
# mlp2 = LeakyReLU()(mlp2)
# mlp2 = Dense(2, activation = 'softmax', name = 'classification_out')(mlp2)

# model = Model(inputs=[main_input], 
#               outputs=[mlp1, mlp2])

    
# reduce_lr = ReduceLROnPlateau(monitor='val_classification_out_loss', factor=0.9, patience=25, min_lr=0.000001, verbose=1)
# checkpointer = ModelCheckpoint(filepath="multitask.hdf5", verbose=1, save_best_only=True)

# opt = Nadam(lr=0.002)

# model.compile(optimizer=opt,
#               loss={'regression_out': 'mape', 'classification_out': 'categorical_crossentropy'},
#               metrics = {'classification_out': ['accuracy']})

# history = model.fit(
#           {'ts_input': X_train},
#           {'regression_out': Y_train1, 
#           'classification_out': Y_train2},
#           validation_data = (X_test, [Y_test1, Y_test2]),
#           epochs=500, batch_size=128, shuffle = True, verbose = True,
#           callbacks=[reduce_lr, checkpointer])

# model.load_weights("multitask.hdf5")

# predicted = model.predict(X_test)
# pred = predicted[1]

# for yt, p in zip(Y_test2, pred):
#     print yt, p

# from sklearn.metrics import classification_report
# from sklearn.metrics import confusion_matrix
# C = confusion_matrix([np.argmax(y) for y in Y_test2], [np.argmax(y) for y in pred])

# print C / C.astype(np.float).sum(axis=1)

# plt.figure()
# plt.plot(history.history['loss'])
# plt.plot(history.history['val_loss'])
# plt.title('model loss')
# plt.ylabel('loss')
# plt.xlabel('epoch')
# plt.legend(['train', 'test'], loc='best')
# plt.show()

# plt.figure()
# plt.plot(history.history['acc'])
# plt.plot(history.history['val_acc'])
# plt.title('model accuracy')
# plt.ylabel('accuracy')
# plt.xlabel('epoch')
# plt.legend(['train', 'test'], loc='best')
# plt.show()
