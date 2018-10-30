from sklearn.ensemble import ExtraTreesRegressor
###XGB Finale - No Outliers
from sklearn import model_selection, preprocessing, ensemble
import pandas as pd
import numpy as np
import datetime as dt
import gc
from sklearn.metrics import mean_absolute_error
from sklearn.externals import joblib

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
yval = y[valindex]
#train = train[trainindex]
#y = y[trainindex]
#################################################

lbound = -0.4#np.mean(y) - 3 * np.std(y)
ubound = 0.419#np.mean(y) + 3 * np.std(y)

test = joblib.load("../input/teststat.pkl")
test = test[cols]
#test = valid # remove
oobval = np.zeros((train.shape[0],1))
oobtest = np.zeros((test.shape[0],1))
valerr = []
val_scores = []
cnt = 0
gc.collect()
nfold =5
for i,n in [(6,900), (10,500), (4,1600)]:
    for seed in [2017]:
        kf = model_selection.KFold(n_splits=nfold, shuffle=True, random_state=seed)    
        for dev_index, val_index in kf.split(train):
            dev_X, val_X = train.values[dev_index], train.values[val_index]
            dev_y, val_y = y[dev_index], y[val_index]
#            dev_X = dev_X[(dev_y >= lbound) & (dev_y <= ubound)]
#            dev_y = dev_y[(dev_y >= lbound) & (dev_y <= ubound)]
            print(dev_X.shape)  

            et = ExtraTreesRegressor(criterion='mse',n_estimators=n,max_features=0.6,max_depth=i,random_state=seed,verbose=0)
            model = et.fit(dev_X, dev_y)
            preds = model.predict(val_X).reshape(-1,1)
            oobval[val_index] += preds
            oobtest += model.predict(test).reshape(-1,1)
            cnt += 1
            valerr.append(mean_absolute_error(val_y, preds))
            print(valerr, "mean:", np.mean(valerr), "std:", np.std(valerr))
            val_scores.append(mean_absolute_error(model.predict(valid), yval))
            print(val_scores, np.mean(val_scores),"---", np.std(val_scores))  
            gc.collect()          
pred2 = oobtest / (3 * nfold)
oobpred2 = oobval / 3
print(mean_absolute_error(y, oobpred2))

######################################################
# Save for L2 Predictors
######################################################
joblib.dump(oobpred2,'../input/train_et_withoutlier.pkl')
joblib.dump(pred2,'../input/test_et_withoutlier.pkl')

