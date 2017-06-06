import pandas as pd
from pandas import Series,DataFrame
import numpy as np
import time
from datetime import date
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk.stem.porter import PorterStemmer

import cPickle

from gensim.models import Word2Vec
import logging

logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')


def load_text_csv(filename = 'Combined_News_DJIA.csv', date_split = date(2014,12,31)):
	'''
		Load news from csv, group them and split in train/test set due to @date_split
	'''
	df = pd.read_csv(filename)
	df['Combined']=df.iloc[:,2:27].apply(lambda row: ''.join(str(row.values)), axis=1)

	train = df.loc[(pd.to_datetime(df["Date"]) <= date_split),['Label','Combined']]
	test = df.loc[(pd.to_datetime(df["Date"]) > date_split),['Label','Combined']]

	return train, test


def load_ts_csv(filename = 'DJIA_table.csv', date_split = date(2014,12,31)):	
	'''
		Load time series from csv, taking adjustment close prices;
		transforming them into percentage of price change;
		split in train/test set due to @date_split
	'''
	data_original = pd.read_csv(filename)[::-1]

	train2 = data_original.loc[(pd.to_datetime(data_original["Date"]) <= date_split)]
	test2 = data_original.loc[(pd.to_datetime(data_original["Date"]) > date_split)]

	data_chng_train = train2.ix[:, 'Adj Close'].pct_change().dropna().tolist()
	data_chng_test = test2.ix[:, 'Adj Close'].pct_change().dropna().tolist()

	return data_chng_train, data_chng_test


def text_process(text):
    '''
    Takes in a string of text, then performs the following:
	    1. Tokenizes and removes punctuation
	    2. Removes  stopwords
	    3. Stems
	    4. Returns a list of the cleaned text
    '''
    if pd.isnull(text):
        return []
    # tokenizing
    tokenizer = RegexpTokenizer(r'\w+')
    text_processed=tokenizer.tokenize(text)
    
    # removing any stopwords
    text_processed = [word.lower() for word in text_processed if word.lower() not in stopwords.words('english')]
    
    # steming
    porter_stemmer = PorterStemmer()
    
    text_processed = [porter_stemmer.stem(word) for word in text_processed]
    
    try:
        text_processed.remove('b')
    except: 
        pass

    return " ".join(text_processed)


def transform_text2sentences(train, test, save_train = 'train_text.p', save_test = 'test_text.p'):
	'''
		Transforming raw text into sentences, 
		if @save_train or @save_test is not None - saves pickles for further use
	'''
	train_text = []
	test_text = []
	for each in train['Combined']:
	    train_text.append(text_process(each))
	for each in test['Combined']:
	    test_text.append(text_process(each))

	if save_train != None: cPickle.dump(train_text, open(save_train, 'wb')) 
	if save_test != None: cPickle.dump(test_text, open(save_test, 'wb')) 

	return train_text, test_text


def transform_text_into_vectors(train_text, test_text, embedding_size = 100, model_path = 'word2vec10.model'):
	'''
		Transforms sentences into sequences of word2vec vectors
		Returns train, test set and trained word2vec model
	'''
	data_for_w2v = []
	for text in train_text + test_text:
	    words = text.split(' ')
	    data_for_w2v.append(words)

	model = Word2Vec(data_for_w2v, size=embedding_size, window=5, min_count=1, workers=4)
	model.save(model_path)
	model = Word2Vec.load(model_path)

	train_text_vectors = [[model[x] for x in sentence.split(' ')] for sentence in train_text]
	test_text_vectors = [[model[x] for x in sentence.split(' ')] for sentence in test_text]

	train_text_vectors = [np.mean(x, axis=0) for x in train_text_vectors]
	test_text_vectors = [np.mean(x, axis=0) for x in test_text_vectors]

	return train_text_vectors, test_text_vectors, model


def split_into_XY(data_chng_train, train_text_vectors, step, window, forecast):
	'''
		Splits textual and time series data into train or test dataset for hybrid model;
		objective y_i is percentage change of price movement for next day
	'''
	X_train, X_train_text, Y_train = [], [], []
	for i in range(0, len(data_chng_train), step): 
	    try:
	        x_i = data_chng_train[i:i+window]
	        y_i = data_chng_train[i+window+forecast]  

	        # text_average = np.mean(train_text_vectors[i:i+WINDOW], axis=0)
	        text_average = train_text_vectors[i:i+window]

	        last_close = x_i[-1]
	        next_close = y_i

	        if y_i > 0.:
	            y_i = [1, 0]
	        else:
	            y_i = [0, 1] 

	    except Exception as e:
	        break

	    X_train.append(x_i)
	    X_train_text.append(text_average)
	    Y_train.append(y_i)

	X_train, X_train_text, Y_train = np.array(X_train), np.array(X_train_text), np.array(Y_train)
	return X_train, X_train_text, Y_train
