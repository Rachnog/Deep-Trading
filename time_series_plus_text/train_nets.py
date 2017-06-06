from process_data import *

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.normalization import BatchNormalization
from keras.layers import Merge
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, CSVLogger, EarlyStopping
from keras.optimizers import RMSprop, Adam, SGD, Nadam
from keras.layers.advanced_activations import *
from keras.layers import Convolution1D, MaxPooling1D, AtrousConvolution1D
from keras.layers.recurrent import LSTM, GRU
from keras import regularizers

#Visualization
import matplotlib.pyplot as plt
import seaborn as sns

import theano
theano.config.compute_test_value = "ignore"


def neural_network(window, embedding):
	'''
		Hybrid neural network for prediction based both on time series and text:
		- model1 takes time series of @window length and passes it into MLP
		- model2 takes averages word2vec vectors from daily news and passes them into LSTM
		- model1 and model2 are merged via other MLP network
	'''
	model1 = Sequential()
	model1.add(Dense(64, input_dim=window,
	                activity_regularizer=regularizers.l2(0.01)))
	model1.add(BatchNormalization())
	model1.add(LeakyReLU())
	model1.add(Dropout(0.75))
	model1.add(Dense(window,
	                activity_regularizer=regularizers.l2(0.01)))
	model1.add(BatchNormalization())
	model1.add(LeakyReLU())


	model2 = Sequential()
	model2.add(LSTM(input_shape = (window, embedding,), output_dim=window, return_sequences=True, recurrent_dropout=0.75))   
	model2.add(LSTM(output_dim=window, return_sequences=False, recurrent_dropout=0.75))                               

	merged = Merge([model1, model2], mode='sum')

	final_model = Sequential()
	final_model.add(merged)

	final_model.add(Dense(16))
	final_model.add(BatchNormalization())
	final_model.add(LeakyReLU())

	final_model.add(Dense(2))
	final_model.add(Activation('softmax'))

	opt = Nadam(lr=0.002)
	final_model.compile(optimizer=opt, 
	              loss='categorical_crossentropy',
	              metrics=['accuracy'])

	return final_model


train, test = load_text_csv()
data_chng_train, data_chng_test = load_ts_csv()

# train_text, test_text = transform_text2sentences(train, test)
train_text = cPickle.load(open('train_text.p', 'rb'))[1:]
test_text = cPickle.load(open('test_text.p', 'rb'))[1:]

train_text_vectors, test_text_vectors, model = transform_text_into_vectors(train_text, test_text, 100)

X_train, X_train_text, Y_train = split_into_XY(data_chng_train, train_text_vectors, 1, 10, 1)
X_test, X_test_text, Y_test = split_into_XY(data_chng_test, test_text_vectors, 1, 10, 1)

final_model = neural_network(10, 100)

# seting callbacks for saving best models and scheduling learning rate
reduce_lr = ReduceLROnPlateau(monitor='val_acc', factor=0.9, patience=50, min_lr=0.000001, verbose=1)
checkpointer = ModelCheckpoint(filepath="test.hdf5", verbose=1, save_best_only=True)

# training the model
history = final_model.fit([X_train, X_train_text], Y_train, 
          nb_epoch = 500, 
          batch_size = 128, 
          verbose=1, 
          validation_data = ([X_test, X_test_text], Y_test),
          callbacks=[reduce_lr, checkpointer],
          shuffle=True)

# plotting performance
plt.figure()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='best')
plt.show()

plt.figure()
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='best')
plt.show()
