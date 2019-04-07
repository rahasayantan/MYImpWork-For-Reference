import cPickle
import os
import sys
import scipy as sc
import operator
import numpy as np
import pandas as pd
from scipy import sparse
import xgboost as xgb
from sklearn import model_selection, preprocessing, ensemble
from sklearn.metrics import log_loss
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from bs4 import BeautifulSoup
from sklearn.metrics import mean_squared_error

#reload(sys)
#sys.setdefaultencoding('utf8')
#r = re.compile(r"\s")
from sklearn.preprocessing import LabelEncoder
import re
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import TruncatedSVD
import nltk
from scipy.stats import boxcox
from sklearn.decomposition import TruncatedSVD
import datetime as dt
from nltk.stem.porter import *
import gc
import math
from collections import Counter

nfold = 5

with open("../pickle05.pkl", "rb") as f:
    (train_df,test_df,train_y,features_to_use,features_to_use,ntrain,test_df_listing_id) = cPickle.load( f)

train_test = pd.concat((train_df, test_df), axis=0).reset_index(drop=True)

###############Model Build and Predict
param = {}
param['objective'] = 'multi:softprob'
param['eta'] = 0.01
param['max_depth'] = 6
param['silent'] = 1
param['num_class'] = 3
param['eval_metric'] = "mlogloss"
param['min_child_weight'] = 1
param['subsample'] = .75
param['colsample_bytree'] = .8
param['seed'] = 12345

###  Ftrs+Desc Ftrs+Ftr Count Vec
features_to_use_ln=[
'listing_id','Zero_building_id', 'Zero_Ftr','Zero_description', 'num_description_words','ratio_description_words', 'num_photos', 'num_features', 
##LOG Price variant
'lg_price',
u'building_id',u'display_address',u'manager_id','bathrooms_cat', 'bedroom_cat','lat_cat','lon_cat',
'manager_id_interest_level_high0','building_id_interest_level_high0',
'manager_id_interest_level_medium0','building_id_interest_level_medium0',
u'manager_id_interest_level_low0',
u'building_id_interest_level_low0',
u'manager_id_interest_level_high_pr0',
u'building_id_interest_level_high_pr0',
u'manager_id_interest_level_medium_pr0',
u'building_id_interest_level_medium_pr0',
u'manager_id_interest_level_low_pr0',
u'building_id_interest_level_low_pr0',
u'manager_id_interest_level_high_bed0',
u'building_id_interest_level_high_bed0',
u'manager_id_interest_level_medium_bed0',
u'building_id_interest_level_medium_bed0',
u'manager_id_interest_level_low_bed0',
u'building_id_interest_level_low_bed0',
u'manager_id_interest_level_high_bath0',
u'building_id_interest_level_high_bath0',
u'manager_id_interest_level_medium_bath0',
u'building_id_interest_level_medium_bath0',
u'manager_id_interest_level_low_bath0',
u'building_id_interest_level_low_bath0'
]


cv_scores = []
bow = CountVectorizer(stop_words='english', max_features=50, ngram_range=(1,1),min_df=3, max_df=.75)
bow.fit(train_test["features_2"])

oob_valpred = np.zeros((train_df.shape[0],3))
oob_tstpred = np.zeros((test_df.shape[0],3))
i=0
kf = model_selection.KFold(n_splits=nfold, shuffle=True, random_state=12345)
for dev_index, val_index in kf.split(range(train_y.shape[0])):
    dev_X, val_X = train_df.iloc[dev_index,:], train_df.iloc[val_index,:]
    dev_y, val_y = train_y[dev_index], train_y[val_index]

    tr_sparse_2 = bow.transform(dev_X["features_2"])
    val_sparse_2 = bow.transform(val_X["features_2"])
    te_sparse_2 = bow.transform(test_df["features_2"])
    
    train_X2 = sparse.hstack([dev_X[features_to_use_ln],tr_sparse_2]).tocsr()#,tr_sparse_d
    val_X2 = sparse.hstack([val_X[features_to_use_ln],val_sparse_2]).tocsr()#,val_sparse_d
    test_X2 = sparse.hstack([test_df[features_to_use_ln], te_sparse_2]).tocsr()
    
    print(train_X2.shape)
    num_rounds =10000
    plst = list(param.items())
    xgtrain = xgb.DMatrix(train_X2, label=dev_y)
    xgval = xgb.DMatrix(val_X2, label=val_y)
    xgtest = xgb.DMatrix(test_X2)

    watchlist = [ (xgtrain,'train'), (xgval, 'val') ]
    model = xgb.train(plst, xgtrain, num_rounds, watchlist, early_stopping_rounds=50)
    best_iteration = model.best_iteration+1
    model = xgb.train(plst, xgtrain, best_iteration, watchlist, early_stopping_rounds=50)
    preds = model.predict(xgval)
    oob_valpred[val_index,...] = preds

    cv_scores.append(log_loss(val_y, preds))
    print(cv_scores)
    print(np.mean(cv_scores))
    print(np.std(cv_scores))

    predtst = model.predict(xgtest)
    oob_tstpred += predtst
oob_tstpred /= nfold
out_df = pd.DataFrame(oob_tstpred)
out_df.columns = ["high", "medium", "low"]
out_df["listing_id"] = test_df_listing_id
out_df.to_csv("../xgb_lblenc_new_ftr_kag3.csv", index=False) 

with open("../xgb_lblenc_new_ftr_kag3.pkl", "wb") as f:
    cPickle.dump((oob_tstpred,oob_valpred), f, -1)

