from sklearn import model_selection, preprocessing, ensemble
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import pandas as pd
import numpy as np
import datetime as dt
import gc
from sklearn.metrics import mean_absolute_error
from sklearn.externals import joblib
from keras.layers.advanced_activations import PReLU
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers.normalization import BatchNormalization
from sklearn.metrics import mean_absolute_error
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding
from keras.optimizers import Adam

train = joblib.load("../input/trainstat.pkl")

cols = [a for a in [y for y in [x for x in [a for a in train.columns if 'X' not in a] if 'countenc0' not in x] if 'meanshftenc0' not in y] if 'yrmonth' not in a]

cols = cols +['X1_Mon_logerror3len',
 'X1_Mon_logerror3mean',
 'X1_Mon_logerror3std',
 'X1_Mon_logerror3skew',
 'X1_Mon_logerror6len',
 'X1_Mon_logerror6mean',
 'X1_Mon_logerror6std',
 'X1_Mon_logerror6skew',
 'X1_Mon_logerror12len',
 'X1_Mon_logerror12mean',
 'X1_Mon_logerror12std',
 'X1_Mon_logerror12skew',
 'X1_Mon_logerror18len',
 'X1_Mon_logerror18mean',
 'X1_Mon_logerror18std',
 'X1_Mon_logerror18skew',
 'X2_Mon_logerror3len',
 'X2_Mon_logerror3mean',
 'X2_Mon_logerror3std',
 'X2_Mon_logerror3skew',
 'X2_Mon_logerror6len',
 'X2_Mon_logerror6mean',
 'X2_Mon_logerror6std',
 'X2_Mon_logerror6skew',
 'X2_Mon_logerror12len',
 'X2_Mon_logerror12mean',
 'X2_Mon_logerror12std',
 'X2_Mon_logerror12skew',
 'X2_Mon_logerror18len',
 'X2_Mon_logerror18mean',
 'X2_Mon_logerror18std',
 'X2_Mon_logerror18skew',
 'X3_Mon_logerror3len',
 'X3_Mon_logerror3mean',
 'X3_Mon_logerror3std',
 'X3_Mon_logerror3skew',
 'X3_Mon_logerror6len',
 'X3_Mon_logerror6mean',
 'X3_Mon_logerror6std',
 'X3_Mon_logerror6skew',
 'X3_Mon_logerror12len',
 'X3_Mon_logerror12mean',
 'X3_Mon_logerror12std',
 'X3_Mon_logerror12skew',
 'X3_Mon_logerror18len',
 'X3_Mon_logerror18mean',
 'X3_Mon_logerror18std',
 'X3_Mon_logerror18skew',
 'X4_Mon_logerror3len',
 'X4_Mon_logerror3mean',
 'X4_Mon_logerror3std',
 'X4_Mon_logerror3skew',
 'X4_Mon_logerror6len',
 'X4_Mon_logerror6mean',
 'X4_Mon_logerror6std',
 'X4_Mon_logerror6skew',
 'X4_Mon_logerror12len',
 'X4_Mon_logerror12mean',
 'X4_Mon_logerror12std',
 'X4_Mon_logerror12skew',
 'X4_Mon_logerror18len',
 'X4_Mon_logerror18mean',
 'X4_Mon_logerror18std',
 'X4_Mon_logerror18skew']


x = train.yrmonth
train = train[cols]
y = joblib.load("../input/y.pkl")
#################################################
# Val Split
#################################################
#x = pd.read_csv('../input/train_2016_v2.csv')
#x["transactiondate"] = pd.to_datetime(x["transactiondate"])
#x["yrmonth"] = x["transactiondate"].apply(lambda x: x.strftime('%Y%m')).astype(int)  

y_logit = x
valindex = y_logit > pd.Period('2017-05')
trainindex = y_logit <= pd.Period('2017-05')
valid = train[valindex]
#train = train[trainindex]
yval = y[valindex]
#y = y[trainindex]
#################################################

lbound = -0.4#np.mean(y) - 3 * np.std(y)
ubound = 0.419#np.mean(y) + 3 * np.std(y)

test = joblib.load("../input/teststat.pkl")
test = test[cols]
#test = valid.copy()

train_test = pd.concat((train, test), axis =0)

for col in train.columns:
    scaler = MinMaxScaler()
    scaler.fit(train_test[col])
    train[col] = scaler.transform(train[col].reshape(-1))
    test[col] = scaler.transform(test[col].reshape(-1))
    valid[col] = scaler.transform(valid[col].reshape(-1))
    del(scaler)
        
        
oobval = np.zeros((train.shape[0],1))
oobtest = np.zeros((test.shape[0],1))
valerr = []
val_scores = []

cnt = 0

def nn_model4():
    model = Sequential()
    model.add(Dense(300, input_dim = train.shape[1], init = 'uniform'))#500
    model.add(PReLU())
    model.add(BatchNormalization())
    model.add(Dropout(0.2))#.2
    model.add(Dense(300,  init = 'uniform'))#500
    model.add(PReLU())
    model.add(BatchNormalization())
    model.add(Dropout(0.2))#.2
    model.add(Dense(200, init = 'uniform'))#400
    model.add(PReLU())
    model.add(BatchNormalization())
    model.add(Dropout(0.2))#.2
    model.add(Dense(200, init = 'uniform'))#400
    model.add(PReLU())
    model.add(BatchNormalization())
    model.add(Dropout(0.1))#.2
    model.add(Dense(1, init='zero'))
    model.compile(loss = 'mean_absolute_error', optimizer = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.))
    return(model)

nbag = 1
nfold = 5

np.random.seed(2017)
for x in np.arange(nbag):
    for seed in [2017]:
        kf = model_selection.KFold(n_splits=nfold, shuffle=True, random_state=seed)    
        for dev_index, val_index in kf.split(y):
            dev_X, val_X = train.values[dev_index], train.values[val_index]
            dev_y, val_y = y[dev_index], y[val_index]

            model = nn_model4()
            earlyStopping=EarlyStopping(monitor='val_loss', patience=20, verbose=1, mode='auto')
            checkpointer = ModelCheckpoint(filepath="./weights2XXLK.hdf5", verbose=1, save_best_only=True)
            fit = model.fit(dev_X,dev_y, batch_size=512, nb_epoch = 100,validation_data=(val_X, val_y),verbose = 1,callbacks=[earlyStopping,checkpointer])
            print("loading weights")
            model.load_weights("./weights2XXLK.hdf5")
            print("predicting..")
            preds = model.predict(val_X)
            oobval[val_index] += preds
            oobtest += model.predict(test.values).reshape(-1,1)
            valerr.append(mean_absolute_error(val_y, preds))
            print(valerr, "mean:", np.mean(valerr), "std:", np.std(valerr))
            val_scores.append(mean_absolute_error(model.predict(valid.values), yval))
            print(val_scores, np.mean(val_scores),"---", np.std(val_scores))            
            
pred1 = oobtest / (nbag * nfold)
oobpred1 = oobval / (nbag )
print(mean_absolute_error(y, oobpred1))

joblib.dump(oobpred1,'../input/train_keras_raw_stat_withoutlier.pkl')
joblib.dump(pred1,'../input/test_keras_raw_stat_withoutlier.pkl')

