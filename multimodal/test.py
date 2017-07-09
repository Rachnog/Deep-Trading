from process_data import *

from keras.models import Sequential
from keras.models import Model
from keras.layers.core import Dense, Dropout, Activation, Flatten, Permute
from keras.layers import Merge, Input, concatenate, average, add
from keras.layers.recurrent import LSTM, GRU
from keras.layers import Convolution1D, MaxPooling1D, AtrousConvolution1D, RepeatVector, AveragePooling1D
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

#Visualization
import matplotlib.pyplot as plt
import seaborn as sns


train, test = load_text_csv()
data_chng_train, data_chng_test = load_ts_csv()

# train_text, test_text = transform_text2sentences(train, test)
train_text = cPickle.load(open('train_text.p', 'rb'))[1:]
test_text = cPickle.load(open('test_text.p', 'rb'))[1:]

train_text_vectors, test_text_vectors, model = transform_text_into_vectors(train_text, test_text, 100)

X_train, X_train_text, Y_train, Y_train2 = split_into_XY(data_chng_train, train_text_vectors, 1, 30, 1)
X_test, X_test_text, Y_test, Y_test2 = split_into_XY(data_chng_test, test_text_vectors, 1, 30, 1)


X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 5))
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 5))


# X_train_text = np.reshape(X_train_text, (X_train_text.shape[0], X_train_text.shape[1], 100))
# X_test_text = np.reshape(X_test_text, (X_test_text.shape[0], X_test_text.shape[1], 100))


main_input = Input(shape=(30, 5), name='ts_input')
text_input = Input(shape=(30, 100), name='text_input')
lstm1 = LSTM(10, return_sequences=True, recurrent_dropout=0.25, dropout=0.25, bias_initializer='ones')(main_input)
lstm1 = LSTM(10, return_sequences=True, recurrent_dropout=0.25, dropout=0.25, bias_initializer='ones')(lstm1)
lstm1 = Flatten()(lstm1)
lstm2 = LSTM(10, return_sequences=True, recurrent_dropout=0.25, dropout=0.25, bias_initializer='ones')(text_input)
lstm2 = LSTM(10, return_sequences=True, recurrent_dropout=0.25, dropout=0.25, bias_initializer='ones')(lstm2)
lstm2 = Flatten()(lstm2)


lstms = concatenate([lstm1, lstm2])


x1 = Dense(64)(lstms)
x1 = LeakyReLU()(x1)
x1 = Dense(1, activation = 'linear', name='regression')(x1)

x2 = Dense(64)(lstms)
x2 = LeakyReLU()(x2)
x2 = Dropout(0.9)(x2)
x2 = Dense(1, activation = 'sigmoid', name = 'class')(x2)

final_model = Model(inputs=[main_input, text_input], 
              outputs=[x1, x2])
opt = Nadam(lr=0.002, clipnorm = 0.5)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.9, patience=50, min_lr=0.000001, verbose=1)
checkpointer = ModelCheckpoint(monitor='val_loss', filepath="model.hdf5", verbose=1, save_best_only=True)
final_model.compile(optimizer=opt, loss={'regression': 'mse', 'class': 'binary_crossentropy'}, loss_weights=[1., 0.2])


for layer in final_model.layers:
    print layer, layer.output_shape


try:
	history = final_model.fit([X_train, X_train_text], [Y_train, Y_train2],
		nb_epoch = 100, 
		batch_size = 256, 
		verbose=1, 
		validation_data=([X_test, X_test_text], [Y_test, Y_test2]), 
		callbacks=[reduce_lr, checkpointer], shuffle=True)
    
	plt.figure()
	plt.plot(history.history['loss'])
	plt.plot(history.history['val_loss'])
	plt.title('model loss')
	plt.ylabel('loss')
	plt.xlabel('epoch')
	plt.legend(['train', 'test'], loc='best')
	plt.show()


except Exception as e:
    print e

finally:
    final_model.load_weights("model.hdf5")
    pred = final_model.predict([X_test, X_test_text])[0]

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
