#Keras embedding

from sklearn.externals import joblib
import scipy as sc
import operator
import numpy as np
import pandas as pd
from scipy import sparse
import xgboost as xgb
from sklearn import model_selection, preprocessing, ensemble
from sklearn.preprocessing import LabelEncoder
from scipy.stats import boxcox
import gc
import math
from collections import Counter
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers.advanced_activations import PReLU
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers.recurrent import LSTM
from keras.layers.normalization import BatchNormalization
from keras.layers.core import Dense, Dropout, Activation, Reshape
from keras.layers.embeddings import Embedding

train_char = joblib.load('../data/train_cat_LblEncode.pkl')
test_char = joblib.load('../data/test_cat_LblEncode.pkl')

def keras_embed_cat(trainX, testX, output_dim = 3):
    model = Sequential()
    model.add(Embedding(np.max(pd.concat((trainX,testX)))+1, output_dim, input_length=1))
    model.compile('rmsprop', 'mse')
    return model.predict(trainX.as_matrix()), model.predict(testX.as_matrix())

lstTrainEmbed = []
lstTestEmbed = []
for col in train_char.columns:
    train_op, test_op = keras_embed_cat(train_char[col], test_char[col], 2)
    lstTrainEmbed.append(train_op)
    lstTestEmbed.append(test_op)
    
#assert train_op.shape == (train_char.shape[0], 1, 3)
train_em = lstTrainEmbed[0].reshape(-1,2)
test_em = lstTestEmbed[0].reshape(-1,2)
for i in np.arange(len(lstTrainEmbed)):
    if len(lstTrainEmbed)-1 == 0:
        break
    train_em = np.hstack((train_em,lstTrainEmbed[i+1].reshape(-1,2)))
    test_em = np.hstack((test_em,lstTestEmbed[i+1].reshape(-1,2)))    
    if (i+1) == len(lstTrainEmbed)-1:
        break

joblib.dump(train_em,'../data/train_cat_kerasEmbed.pkl')
joblib.dump(test_em,'../data/test_cat_kerasEmbed.pkl')

train_num = joblib.load('../data/train_num_wo_corr=1_wholeData.pkl')
test_num = joblib.load('../data/test_num_wo_corr=1_wholeData.pkl')

lstTrainEmbed = []
lstTestEmbed = []
for col in train_num.columns:
    train_op, test_op = keras_embed_cat(train_num[col], test_num[col], 1)
    lstTrainEmbed.append(train_op)
    lstTestEmbed.append(test_op)
    
#assert train_op.shape == (train_char.shape[0], 1, 3)
train_em = lstTrainEmbed[0].reshape(-1,1)
test_em = lstTestEmbed[0].reshape(-1,1)
for i in np.arange(len(lstTrainEmbed)):
    if len(lstTrainEmbed)-1 == 0:
        break
    train_em = np.hstack((train_em,lstTrainEmbed[i+1].reshape(-1,1)))
    test_em = np.hstack((test_em,lstTestEmbed[i+1].reshape(-1,1)))    
    if (i+1) == len(lstTrainEmbed)-1:
        break

joblib.dump(train_em,'../data/train_num_kerasEmbed.pkl')
joblib.dump(test_em,'../data/test_num_kerasEmbed.pkl')

