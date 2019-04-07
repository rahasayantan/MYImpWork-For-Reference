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
nbag = 10
with open("../pickle03.pkl", "rb") as f:
    (train_df,test_df,train_y,features_to_use,features_to_use_ln,ntrain,test_df_listing_id) = cPickle.load( f)

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
param['subsample'] = .9
param['colsample_bytree'] = .8
param['seed'] = 12345

###  Ftrs+Desc Ftrs+Ftr Count Vec
features_to_use_ln=[
'listing_id','Zero_building_id', 'Zero_Ftr','Zero_description', 'num_description_words','ratio_description_words', 'num_photos', 'num_features', 'top_1_manager', 'top_2_manager','top_5_manager', 'top_10_manager', 'top_15_manager','top_20_manager', 'top_25_manager', 'top_30_manager','top_50_manager', 'bottom_10_manager', 'bottom_20_manager','bottom_30_manager', 'top_1_building', 'top_2_building','top_5_building', 'top_10_building', 'top_15_building','top_20_building', 'top_25_building', 'top_30_building','top_50_building', 'bottom_10_building', 'bottom_20_building','bottom_30_building', 'top_1_add', 'top_2_add', 'top_5_add','top_10_add', 'top_15_add', 'top_20_add', 'top_25_add','top_30_add', 'top_50_add', 'bottom_10_add', 'bottom_20_add','bottom_30_add',
##LOG Price variant
'lg_price','per_bed_price','per_bath_price','per_bed_price_dev','per_bath_price_dev', #'lg_price_rnd',
##BoxCox Price variant
#'bc_price','per_bed_price_bc','per_bath_price_bc','per_bed_price_dev_bc','per_bath_price_dev_bc',#bc_price_rnd,
###label encoding
u'building_id', u'created',u'display_address',  u'manager_id', u'street_address','created_year', 'created_month','created_day', 'created_hour', 'created_weekday', 'created_wd','bed_bath','street', 'avenue', 'east', 'west', 'north','south', 'other_address', 'bathrooms_cat', 'bedroom_cat','lat_cat','lon_cat', #'lat_cat_rnd','lon_cat_rnd'#,
'per_bed_bath_price','bedPerBath','bedBathDiff','bedBathSum','bedsPerc','per_bed_price_rat','per_bath_price_rat','manager_id_interest_level_high0','building_id_interest_level_high0','manager_id_interest_level_medium0','building_id_interest_level_medium0'
]


cv_scores = []
bow = CountVectorizer(stop_words='english', max_features=100, ngram_range=(1,1),min_df=2, max_df=.85)
bow.fit(train_test["features_2"])

oob_valpred = np.zeros((train_df.shape[0],3))
oob_tstpred = np.zeros((test_df.shape[0],3))
i=0

with open("../xgb_lblenc_ftrcntvecraw_newftr_lgprice.pkl", "rb") as f:
    (x1,y1) = cPickle.load( f)
with open("../xgb_lblenc_ftrcntvec200_lgprice.pkl", "rb") as f:
    (x2,y2) = cPickle.load( f)
with open("../xgb_lblenc_ftrcntvec200_deskpi_lgprice.pkl", "rb") as f:
    (x3,y3) = cPickle.load( f)
with open("../xgb_lblenc_ftrcntvec200_bcprice.pkl", "rb") as f:
    (x4,y4) = cPickle.load( f)
with open("../xgb_cntenc_ftrcntvec200_lnprice.pkl", "rb") as f:
    (x5,y5) = cPickle.load( f)
with open("../xgb_lblenc_ftrcntvec200_deskpi_rnd_bcprice.pkl", "rb") as f:
    (x6,y6) = cPickle.load( f)
with open("../xgb_tgtenc_ftrcntvec200_bcprice.pkl", "rb") as f:
    (x7,y7) = cPickle.load( f)
with open("../xgb_rnkenc_ftrcntvec200_bcprice.pkl", "rb") as f:
    (x8,y8) = cPickle.load( f)
with open("../xgb_reg_rmse_lblenc_ftrcntvec200_lgprice.pkl", "rb") as f:
    (x9,y9) = cPickle.load( f)
with open("../xgb_poi_lblenc_ftrcntvec200_lgprice.pkl", "rb") as f:
    (x10,y10) = cPickle.load( f)
with open("../xgb_reg_rmse_lblenc_ftrcntvec200_lgprice_newftr.pkl", "rb") as f:
    (x11,y11) = cPickle.load( f)
with open("../xgb_poi_lblenc_ftrcntvec200_lgprice_newftr.pkl", "rb") as f:
    (x12,y12) = cPickle.load( f)
with open("../xgb_lblenc_ftrcntvec200_desctfidf_descmtr_lgprice.pkl", "rb") as f:
    (x16,y16) = cPickle.load( f)
with open("../et_lblenc_ftrcntvecraw_newftr_bcprice_100.pkl", "rb") as f:
    (x13,y13) = cPickle.load( f)
with open("../xgb_restack_l2_50ftr_pred.pkl", "rb") as f:
    (x14,y14) = cPickle.load( f)
with open("../xgb_restack_l2_regre50_pred.pkl", "rb") as f:
    (x18,y18) = cPickle.load( f)
with open("../xgb_restack_l2_woftr_wolisting_rnd_pred.pkl", "rb") as f:
    (x19,y19) = cPickle.load( f)
with open("../xgb_restack_l2_50ftr_rnd_pred.pkl", "rb") as f:
    (x20,y20) = cPickle.load( f)
with open("../keras_minMax_targetenc_200Ftr.pkl", "rb") as f:
    (x21,y21) = cPickle.load( f)
with open("../keras_minMax_cnt_50Ftr.pkl", "rb") as f:
    (x22,y22) = cPickle.load( f)
with open("../keras_regre_minMax_targetenc_200Ftr.pkl", "rb") as f:
    (x23,y23) = cPickle.load( f)
with open("../et-lbl-ftr-cvecraw-newftr-bc-10.pkl", "rb") as f:
    (x24,y24) = cPickle.load( f)
with open("../ada_lblenc_ftrcntvecraw_newftr_bcprice_50.pkl", "rb") as f:
    (x15,y15) = cPickle.load( f)
    
test_df2 = np.hstack((x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11,x12,x16,x13,x14,x18,x19,x20,x21,x22,x23,x24,x15,test_df[features_to_use_ln].values))
train_df2 = np.hstack((y1,y2,y3,y4,y5,y6,y7,y8,y9,y10,y11,y12,y16,y13,y14,y18,y19,y20,y21[:49352,:],y22[:49352,:],y23[:49352,:],y24,y15,train_df[features_to_use_ln].values))

for x in np.arange(nbag):
    kf = model_selection.KFold(n_splits=nfold, shuffle=True, random_state=12345*x)    
    for dev_index, val_index in kf.split(range(train_y.shape[0])):
        dev_X, val_X = train_df2[dev_index,:], train_df2[val_index,:]
        dev_y, val_y = train_y[dev_index], train_y[val_index]

        tr_sparse_2 = bow.transform(train_df.loc[dev_index,"features_2"])
        val_sparse_2 = bow.transform(train_df.loc[val_index,"features_2"])
        te_sparse_2 = bow.transform(test_df["features_2"])
        
        train_X2 = sparse.hstack([dev_X,tr_sparse_2]).tocsr()#,tr_sparse_d
        val_X2 = sparse.hstack([val_X,val_sparse_2]).tocsr()#,val_sparse_d
        test_X2 = sparse.hstack([test_df2, te_sparse_2]).tocsr()

        print(train_X2.shape)
        print(test_X2.shape)

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
        oob_valpred[val_index,...] += preds

        cv_scores.append(log_loss(val_y, preds))
        print(cv_scores)
        print(np.mean(cv_scores))
        print(np.std(cv_scores))

        predtst = model.predict(xgtest)
        oob_tstpred += predtst
oob_valpred /=nbag
oob_tstpred /= (nfold*nbag)
out_df = pd.DataFrame(oob_tstpred)#
out_df.columns = ["high", "medium", "low"]
out_df["listing_id"] = test_df_listing_id
out_df.to_csv("../xgb_restack_l2_pred.csv", index=False) 

#######################
############Keras
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers.advanced_activations import PReLU
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers.recurrent import LSTM
from keras.layers.normalization import BatchNormalization
from keras.layers.core import Dense, Dropout, Activation, Merge, Reshape
from keras.layers.embeddings import Embedding

def nn_model4():
    model = Sequential()
    model.add(Dense(100, input_dim = train_X2.shape[1], init = 'uniform'))#500
    model.add(PReLU())
    model.add(BatchNormalization())
    model.add(Dropout(0.3))#.2
    model.add(Dense(100, init = 'uniform'))#400
    model.add(PReLU())
    model.add(BatchNormalization())
    model.add(Dropout(0.3))#.2
    model.add(Dense(3, init='zero'))
    model.add(Activation('softmax'))##
    model.compile(loss = 'categorical_crossentropy', optimizer = 'adam')
    return(model)

with open("../xgb_lblenc_ftrcntvecraw_newftr_lgprice.pkl", "rb") as f:
    (x1,y1) = cPickle.load( f)
with open("../xgb_lblenc_ftrcntvec200_lgprice.pkl", "rb") as f:
    (x2,y2) = cPickle.load( f)
with open("../xgb_lblenc_ftrcntvec200_deskpi_lgprice.pkl", "rb") as f:
    (x3,y3) = cPickle.load( f)
with open("../xgb_lblenc_ftrcntvec200_bcprice.pkl", "rb") as f:
    (x4,y4) = cPickle.load( f)
with open("../xgb_cntenc_ftrcntvec200_lnprice.pkl", "rb") as f:
    (x5,y5) = cPickle.load( f)
with open("../xgb_lblenc_ftrcntvec200_deskpi_rnd_bcprice.pkl", "rb") as f:
    (x6,y6) = cPickle.load( f)
with open("../xgb_tgtenc_ftrcntvec200_bcprice.pkl", "rb") as f:
    (x7,y7) = cPickle.load( f)
with open("../xgb_rnkenc_ftrcntvec200_bcprice.pkl", "rb") as f:
    (x8,y8) = cPickle.load( f)
with open("../xgb_reg_rmse_lblenc_ftrcntvec200_lgprice.pkl", "rb") as f:
    (x9,y9) = cPickle.load( f)
with open("../xgb_poi_lblenc_ftrcntvec200_lgprice.pkl", "rb") as f:
    (x10,y10) = cPickle.load( f)
with open("../xgb_reg_rmse_lblenc_ftrcntvec200_lgprice_newftr.pkl", "rb") as f:
    (x11,y11) = cPickle.load( f)
with open("../xgb_poi_lblenc_ftrcntvec200_lgprice_newftr.pkl", "rb") as f:
    (x12,y12) = cPickle.load( f)
with open("../xgb_lblenc_ftrcntvec200_desctfidf_descmtr_lgprice.pkl", "rb") as f:
    (x16,y16) = cPickle.load( f)
with open("../et_lblenc_ftrcntvecraw_newftr_bcprice_100.pkl", "rb") as f:
    (x13,y13) = cPickle.load( f)
with open("../xgb_restack_l2_50ftr_pred.pkl", "rb") as f:
    (x14,y14) = cPickle.load( f)
with open("../xgb_restack_l2_regre50_pred.pkl", "rb") as f:
    (x18,y18) = cPickle.load( f)
with open("../xgb_restack_l2_woftr_wolisting_rnd_pred.pkl", "rb") as f:
    (x19,y19) = cPickle.load( f)
with open("../xgb_restack_l2_50ftr_rnd_pred.pkl", "rb") as f:
    (x20,y20) = cPickle.load( f)
with open("../keras_minMax_targetenc_200Ftr.pkl", "rb") as f:
    (x21,y21) = cPickle.load( f)
with open("../keras_minMax_cnt_50Ftr.pkl", "rb") as f:
    (x22,y22) = cPickle.load( f)
with open("../keras_regre_minMax_targetenc_200Ftr.pkl", "rb") as f:
    (x23,y23) = cPickle.load( f)
with open("../et-lbl-ftr-cvecraw-newftr-bc-10.pkl", "rb") as f:
    (x24,y24) = cPickle.load( f)
with open("../ada_lblenc_ftrcntvecraw_newftr_bcprice_50.pkl", "rb") as f:
    (x15,y15) = cPickle.load( f)
with open("../xgb_lblenc_lgprice_fewFTR.pkl", "rb") as f:
    (x25,y25) = cPickle.load( f)
with open("../xgb_few_ftrs.pkl", "rb") as f:
    (x26,y26) = cPickle.load( f)
with open("../xgb_listing_id.pkl", "rb") as f:
    (x27,y27) = cPickle.load( f)
with open("../xgb_ftr_desc.pkl", "rb") as f:
    (x28,y28) = cPickle.load( f)
    
test_df2 = np.hstack((x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11,x12,x16,x13,x14,x18,x19,x20,x21,x22,x23,x24,x15,x25,x26,x27,x28))
train_df2 = np.hstack((y1,y2,y3,y4,y5,y6,y7,y8,y9,y10,y11,y12,y16,y13,y14,y18,y19,y20,y21[:49352,:],y22[:49352,:],y23[:49352,:],y24,y15,y25,y26,y27,y28))
    
cv_scores = []
oob_valpred = np.zeros((train_df.shape[0],3))
oob_tstpred = np.zeros((test_df.shape[0],3))
train_y2 = np_utils.to_categorical(train_y, 3)
for x in np.arange(nbag):
    kf = model_selection.KFold(n_splits=nfold, shuffle=True, random_state=12345*x)    
    for dev_index, val_index in kf.split(range(train_y.shape[0])):
        train_X2, val_X2 = train_df2[dev_index,:], train_df2[val_index,:]
        dev_y, val_y = train_y2[dev_index], train_y2[val_index]
        test_X2 = test_df2.copy()
        print(train_X2.shape)
        
        model = nn_model4()
        earlyStopping=EarlyStopping(monitor='val_loss', patience=50, verbose=1, mode='auto')
        checkpointer = ModelCheckpoint(filepath="./weights2XXLK.hdf5", verbose=1, save_best_only=True)
        fit = model.fit(train_X2, dev_y, 
                                  nb_epoch = 10000,
                                  validation_data=(val_X2, val_y),
                                  verbose = 1,callbacks=[earlyStopping,checkpointer]
                                  )
        print("loading weights")
        model.load_weights("./weights2XXLK.hdf5")
        print("predicting..")
        
        preds = model.predict(val_X2)#[:,0]
        oob_valpred[val_index,...] += preds
        cv_scores.append(log_loss(val_y, preds))
        print(cv_scores)
        print(np.mean(cv_scores))
        print(np.std(cv_scores))

        predtst = (model.predict(test_X2))#[:,0]
        oob_tstpred += predtst
oob_valpred /= nbag
oob_tstpred /= (nfold*nbag)
out_df = pd.DataFrame(oob_tstpred)
out_df.columns = ["high", "medium", "low"]
out_df["listing_id"] = test_df_listing_id
out_df.to_csv("../keras_L2.csv", index=False) 

with open("../keras_L2.pkl", "wb") as f:
    cPickle.dump((oob_tstpred,oob_valpred), f, -1)
###Old Score
#[0.52305209635321348, 0.51907342921080069, 0.52102132207204954, 0.5201797693216722, 0.51651091318463827]
#0.519967506028
#0.00216414827934
#New Score
#[0.5228894522984826, 0.51887473053048139, 0.52087177150944586, 0.52010859504893847, 0.51494352591063364]
#0.51953761506
#0.00264143428707
############Combine
testIdSTCKNET = pd.read_csv("../stacknet/test_stacknet.csv",usecols=[0],header=None)
out_df3 = pd.read_csv("../stacknet/sigma_stack_pred_restack.csv",header=None)#../stacknet/submission_0.538820662797.csv -non restacking
out_df3 = pd.concat([testIdSTCKNET,out_df3],axis=1)

out_df1 = pd.read_csv("../xgb_restack_l2_pred.csv")
out_df2 = pd.read_csv("../keras_L2.csv")
out_df2.columns =["high2", "medium2", "low2","listing_id"]
#out_df3 = pd.read_csv("../stacknet/submission_0.538820662797.csv")#../stacknet/submission_0.538820662797.csv -non restacking
out_df3.columns =["listing_id","high3", "medium3", "low3"]

out_df_fin = out_df1.merge(out_df2, how="left", on="listing_id").merge(out_df3, how="left", on="listing_id")
#out_df_fin["high"] = 0.33*out_df_fin["high"]+0.33*out_df_fin["high2"]+0.34*out_df_fin["high3"]
#out_df_fin["medium"] = 0.33*out_df_fin["medium"]+0.33*out_df_fin["medium2"]+0.34*out_df_fin["medium3"]
#out_df_fin["low"] = 0.33*out_df_fin["low"]+0.33*out_df_fin["low2"]+0.34*out_df_fin["low3"]
#out_df_fin2 = out_df_fin[out_df1.columns]
#out_df_fin2.to_csv("../L2_stk_restk.csv", index=False) 

out_df_fin["high"] = 0.3*out_df_fin["high"]+0.5*out_df_fin["high2"]+0.2*out_df_fin["high3"]
out_df_fin["medium"] = 0.3*out_df_fin["medium"]+0.5*out_df_fin["medium2"]+0.2*out_df_fin["medium3"]
out_df_fin["low"] = 0.3*out_df_fin["low"]+0.5*out_df_fin["low2"]+0.2*out_df_fin["low3"]
out_df_fin2 = out_df_fin[out_df1.columns]
out_df_fin2.to_csv("../L2_stk_restk.csv", index=False) 


###########################
#out_df1 = pd.read_csv("../keras_L2.csv")
#out_df3 = pd.read_csv("../stacknet/submission_0.538820662797.csv")
#out_df3.columns =["listing_id","high3", "medium3", "low3"]
#out_df_fin = out_df1.merge(out_df3, how="left", on="listing_id")
#out_df_fin["high"] = 0.5*out_df_fin["high"]+0.5*out_df_fin["high3"]
#out_df_fin["medium"] = 0.5*out_df_fin["medium"]+0.5*out_df_fin["medium3"]
#out_df_fin["low"] = 0.5*out_df_fin["low"]+0.5*out_df_fin["low3"]
#out_df_fin2 = out_df_fin[out_df1.columns]
#out_df_fin2.to_csv("../best_add_st.csv", index=False) 
############################
#from matplotlib import pylab as plt
#
#def create_feature_map(features):
#    outfile = open('xgb.fmap', 'w')
#    i = 0
#    for feat in features:
#        outfile.write('{0}\t{1}\tq\n'.format(i, feat))
#        i = i + 1
#
#    outfile.close()
#    
#create_feature_map(features_to_use_ln)
#importance = model.get_fscore(fmap='xgb.fmap')
#importance = sorted(importance.items(), key=operator.itemgetter(1))
#
#df = pd.DataFrame(importance, columns=['feature', 'fscore'])
#df['fscore'] = df['fscore'] / df['fscore'].sum()
#
#plt.figure()
#df.plot()
#df.plot(kind='barh', x='feature', y='fscore', legend=False, figsize=(6, 10))
#plt.title('XGBoost Feature Importance')
#plt.xlabel('relative importance')
#plt.gcf().savefig('feature_importance_xgb.png')    
#
#df.sort(['feature'],ascending=False)
#df