import numpy as np
import datetime as dt
import time
import pandas as pd

from utils import *
from indicators import *


class OHLCFeatureExtractor():
	'''
		Basic class for loading OHLC(V) time series and splitting 
		them into X and Y (features and targets)
	'''

	def __init__(self):
		self.timeseries = None
		self.X = None
		self.Y = None


	def fit(self, data):
		self.dataframe = data
		self.date = pd.to_datetime(data.ix[:, 'Date'])
		self.data_week = self.date.dt.dayofweek.values
		self.data_year = self.date.dt.dayofyear.values

		self.open_price = data.ix[:, 'Open']
		self.high_price = data.ix[:, 'High']
		self.low_price = data.ix[:, 'Low']
		self.close_price = data.ix[:, 'Close']
		self.volume = data.ix[:, 'Volume']

		self.basic_names = data.columns.values.tolist()


	def get_close_prices(self, time_window, predict_horizon, step=1, binary = False, ternary = True):
		self.timeseries = np.array(self.close_price)
		return self.split_into_XY(time_window, predict_horizon, step=step, ternary=ternary)


	def get_ohlc(self, time_window, predict_horizon, step=1, binary=True):
		self.timeseries = np.column_stack((self.open_price.tolist(),
							   self.high_price.tolist(),
							   self.low_price.tolist(),
							   self.close_price.tolist()))
		return self.split_into_XY_multidimensional(3, time_window, predict_horizon, step=step, binary=binary) # TODO fix magic number 3


	def get_ohlcv(self, time_window, predict_horizon, step=1, binary=True):
		self.timeseries = np.column_stack((self.open_price.tolist(),
							   self.high_price.tolist(),
							   self.low_price.tolist(),
							   self.close_price.tolist(),
							   self.volume.tolist()))
		return self.split_into_XY_multidimensional(3, time_window, predict_horizon, step=step, binary=binary) # TODO fix magic number 3


	def get_ohlcvwy(self, time_window, predict_horizon, step=1, binary=True):
		self.timeseries = np.column_stack((self.open_price.tolist(),
							   self.high_price.tolist(),
							   self.low_price.tolist(),
							   self.close_price.tolist(),
							   self.volume.tolist(),
							   self.data_week.tolist(),
							   self.data_year.tolist()))
		return self.split_into_XY_multidimensional(3, time_window, predict_horizon, step=step, binary=binary) # TODO fix magic number 3



	def get_close_prices_difference(self, time_window, predict_horizon, step=1, pct = True, binary=False):
		if pct:
			self.timeseries = np.array(self.close_price.pct_change())
		else:
			self.timeseries = np.array(self.close_price.diff())

		self.timeseries = remove_nan_examples(self.timeseries)
		return self.split_into_XY(time_window, predict_horizon, step=step, binary=binary, diff=True)


	def get_all_columns_difference(self, time_window, predict_horizon, skip_columns = ['Date', 'Adj Close'], step=1, pct = False, binary=True):
		channels = ()
		for column_name in self.basic_names:
			if column_name not in skip_columns:
				if pct:
					ch = self.dataframe.ix[:, column_name].pct_change().tolist()
				else:
					ch = self.dataframe.ix[:, column_name].diff().tolist()			
				channels += (ch, )
		self.timeseries = np.column_stack(channels)
		self.timeseries = remove_nan_examples(self.timeseries)
		return self.split_into_XY_multidimensional(3, time_window, predict_horizon, step=step, binary=binary) # TODO fix magic number 3


	def set_dataframe(self, dataframe):
		self.dataframe = dataframe
		self.fit(dataframe)


	def get_all_columns(self, time_window, predict_horizon, skip_columns = ['Date', 'Adj Close'], step=1, binary=True):
		channels = ()
		for column_name in self.basic_names:
			if column_name not in skip_columns:
				ch = self.dataframe.ix[:, column_name].tolist()
				channels += (ch, )
		self.timeseries = np.column_stack(channels)
		self.timeseries = remove_nan_examples(self.timeseries)
		return self.split_into_XY_multidimensional(3, time_window, predict_horizon, step=step, binary=binary) # TODO fix magic number 3


	def split_into_XY(self, train, predict, step, binary=False, ternary=False, diff=False):
		'''
			time_window - historical time window on which base we predict the future
			predict_horizon - on which range in the future we want to predict
			step - how many time stamps we can skip while generating training set (1 = no skip)
			binary - True if generate binary features:
					[1, 0] if price will go up
					[0, 1] if price will go down
					 False, if predict real value of prediction horizon
		'''

		print binary, ternary, diff

		data = self.timeseries
		X, Y = [], []
		for i in range(0, len(data), step):
		    try:
		        x_i = data[i:i+train]
		        y_i = data[i+train+predict]  

		        last_close = x_i[train-1]
		        next_close = y_i

		        if binary:
			        if last_close < next_close:
			            y_i = 1.
			        else:
			            y_i = 0.      

				if diff:
					if next_close > 0:
						y_i = 1.
					else:
						y_i = 0.   

				if ternary:
					if abs(1. - last_close/next_close) > 0.1:
						y_i = 1.
					elif abs(1. - last_close/next_close) < 0.1:
						y_i = -1.
					else:
						y_i = 0.

		    except Exception as e:
		        break

		    X.append(x_i)
		    Y.append(y_i)

		return np.array(X), np.array(Y)


	def split_into_XY_multidimensional(self, target_index, train, predict, step, binary=True):
		'''
			target_index - index of close price (or other value we want to predict in multi-dim array)
			time_window - historical time window on which base we predict the future
			predict_horizon - on which range in the future we want to predict
			step - how many time stamps we can skip while generating training set (1 = no skip)
			binary - True if generate binary features:
					[1, 0] if price will go up
					[0, 1] if price will go down
					 False, if predict real value of prediction horizon
		'''
		data = self.timeseries
		X, Y = [], []
		for i in range(0, len(data), step):
		    try:
		        x_i = data[i:i+train]
		        y_i = data[i+train+predict]  

		        last_close = x_i[train-1][target_index]
		        next_close = y_i[target_index]
		        y_i = next_close

		        if binary:
			        if last_close < next_close:
			            y_i = 1.
			        else:
			            y_i = 0.   

		    except:
		        break

		    X.append(x_i)
		    Y.append(y_i)

		return np.array(X), np.array(Y)