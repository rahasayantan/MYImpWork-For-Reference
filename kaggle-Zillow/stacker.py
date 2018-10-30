import pandas as pd
import numpy as np
from sklearn import model_selection, preprocessing, ensemble
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.externals import joblib

train1 = joblib.load('../input/train_xgb_meanshft_stat_enc_wo.pkl')
test1 = joblib.load('../input/test_xgb_meanshft_stat_enc_wo.pkl')

train2 = joblib.load('../input/train_xgb_meanshft_enc_wo.pkl')
test2 = joblib.load('../input/test_xgb_meanshft_enc_wo.pkl')

train3 = joblib.load('../input/train_xgb_lr_wo.pkl')
test3 = joblib.load('../input/test_xgb_lr_wo.pkl')

train4 = joblib.load('../input/train_xgb_count_enc_wo.pkl')
test4 = joblib.load('../input/test_xgb_count_enc_wo.pkl')

train5 = joblib.load('../input/train_rgf_cntenc.pkl')
test5 = joblib.load('../input/test_rgf_cntenc.pkl')

train6 = joblib.load('../input/train_keras_enc_stat.pkl')
test6 = joblib.load('../input/test_keras_enc_stat.pkl')

train7 = joblib.load('../input/train_lgb_raw_stat_withoutliers.pkl')
test7 = joblib.load('../input/test_lgb_raw_stat_withoutliers.pkl')

train8 = joblib.load('../input/train_keras_raw_stat_withoutlier.pkl')
test8 = joblib.load('../input/test_keras_raw_stat_withoutlier.pkl')

train9 = joblib.load('../input/train_keras_raw_stat_withoutlier.pkl')
test9 = joblib.load('../input/test_keras_raw_stat_withoutlier.pkl')

train10 = joblib.load('../input/train_et_withoutlier.pkl')
test10 = joblib.load('../input/test_et_withoutlier.pkl')

train11 = joblib.load('../input/train_lasso_withoutlier.pkl')
test11 = joblib.load('../input/test_lasso_withoutlier.pkl')

train12 = joblib.load('../input/train_knn_withoutlier.pkl')
test12 = joblib.load('../input/test_knn_withoutlier.pkl')

y = joblib.load("../input/y.pkl")

###############SPLIT
train = joblib.load("../input/trainstat.pkl")
x = train.yrmonth

y_logit = x
valindex = y_logit > pd.Period('2017-05')
trainindex = y_logit <= pd.Period('2017-05')
valid = train[valindex]
train = train[trainindex]
yval = y[valindex]
y = y[trainindex]
#################################################
lbound = -0.4#np.mean(y) - 3 * np.std(y)
ubound = 0.419#np.mean(y) + 3 * np.std(y)

#################################################
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers.advanced_activations import PReLU
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers.recurrent import LSTM
from keras.layers.normalization import BatchNormalization
from keras.layers.core import Dense, Dropout, Activation, Reshape
from keras.layers.embeddings import Embedding
from keras.optimizers import Adam
def nn_model4():
    model = Sequential()
    model.add(Dense(50, input_dim = train.shape[1], init = 'uniform'))#500
    model.add(PReLU())
    model.add(BatchNormalization())
    model.add(Dropout(0.1))#.2
    model.add(Dense(50, init = 'uniform'))#400
    model.add(PReLU())
    model.add(BatchNormalization())
    model.add(Dropout(0.1))#.2
    model.add(Dense(1, init='zero'))
    model.compile(loss = 'mean_absolute_error', optimizer = Adam(lr=0.00001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.))
#)optimizer = 'adam')
    return(model)

train = np.hstack((train1, train2, train3, train4, train5, train6, train7, train8, train9, train10, train11, train12))
test = np.hstack((test1, test2, test3, test4, test5, test6, test7, test8, test9, test10, test11, test12))


oob_tstpred = np.zeros((test.shape[0],1))
oobval = np.zeros((train.shape[0],1))
cv_scores = []
nbag, nfold = 1, 5

np.random.seed(2017)

for x in np.arange(nbag):
    kf = model_selection.KFold(n_splits=nfold, shuffle=True, random_state=12345)    
    for dev_index, val_index in kf.split(train):
        dev_X, val_X = train[dev_index,:], train[val_index,:]
        dev_y, val_y = y[dev_index], y[val_index]
        dev_X = dev_X[(dev_y >= lbound) & (dev_y <= ubound)]
        dev_y = dev_y[(dev_y >= lbound) & (dev_y <= ubound)]
        val_X2 = val_X[(val_y >= lbound) & (val_y <= ubound)]
        val_y2 = val_y[(val_y >= lbound) & (val_y <= ubound)]
        
        model = nn_model4()
        earlyStopping=EarlyStopping(monitor='val_loss', patience=50, verbose=1, mode='auto')
        checkpointer = ModelCheckpoint(filepath="./weights2XXLK.hdf5", verbose=1, save_best_only=True)
        fit = model.fit(dev_X, dev_y, 
                                  nb_epoch = 100,
                                  validation_data=(val_X2, val_y2),
                                  verbose = 1,callbacks=[earlyStopping,checkpointer]
                                  )
        print("loading weights")
        model.load_weights("./weights2XXLK.hdf5")
        print("predicting..")

        preds = model.predict(val_X)#[:,0]
        oobval[val_index] += preds.reshape(-1,1)
        cv_scores.append(mean_absolute_error(val_y, preds))
        print(cv_scores)
        print(np.mean(cv_scores))
        print(np.std(cv_scores))

        predtst = (model.predict(test))#[:,0]
        oob_tstpred += predtst

oob_tstpredx = oob_tstpred/(nfold*nbag)
oobvalx = oobval/nbag

val1 = oobvalx.copy()
test1 = oob_tstpredx.cpy()
########################################
# With Outliers
oob_tstpred = np.zeros((test.shape[0],1))
oobval = np.zeros((train.shape[0],1))
cv_scores = []
nbag, nfold = 1, 5

np.random.seed(2017)

for x in np.arange(nbag):
    kf = model_selection.KFold(n_splits=nfold, shuffle=True, random_state=12345)    
    for dev_index, val_index in kf.split(train):
        dev_X, val_X = train[dev_index,:], train[val_index,:]
        dev_y, val_y = y[dev_index], y[val_index]
        
        model = nn_model4()
        earlyStopping=EarlyStopping(monitor='val_loss', patience=50, verbose=1, mode='auto')
        checkpointer = ModelCheckpoint(filepath="./weights2XXLK.hdf5", verbose=1, save_best_only=True)
        fit = model.fit(dev_X, dev_y, 
                                  nb_epoch = 100,
                                  validation_data=(val_X, val_y),
                                  verbose = 1,callbacks=[earlyStopping,checkpointer]
                                  )
        print("loading weights")
        model.load_weights("./weights2XXLK.hdf5")
        print("predicting..")

        preds = model.predict(val_X)#[:,0]
        oobval[val_index] += preds.reshape(-1,1)
        cv_scores.append(mean_absolute_error(val_y, preds))
        print(cv_scores)
        print(np.mean(cv_scores))
        print(np.std(cv_scores))

        predtst = (model.predict(test))#[:,0]
        oob_tstpred += predtst

oob_tstpredx = oob_tstpred/(nfold*nbag)
oobvalx = oobval/nbag


######

finalpred = oob_tstpredx
oobfinalpred = oobvalx

testparcelid = joblib.load('../input/testparcel.pkl')

output = pd.DataFrame({'ParcelId': testparcelid.astype(np.int32).reshape(-1),
        '201610': finalpred.reshape(-1), '201611': finalpred.reshape(-1), '201612': finalpred.reshape(-1),
        '201710': finalpred.reshape(-1), '201711': finalpred.reshape(-1), '201712': finalpred.reshape(-1)})

output.to_csv('../results/kerasl2.csv.gz',compression='gzip', index=False, float_format='%.4g')  

