###XGB Finale - No Outliers
from sklearn import model_selection
import xgboost as xgb
import pandas as pd
import numpy as np
import datetime as dt
import gc
from sklearn.metrics import mean_absolute_error
from sklearn.externals import joblib

##########################################################
# Mean Encode
##########################################################
cols = ['resell','latitude', 'countnullcol', 'yearbuilt', 'longitude',
		'taxamount','calculatedfinishedsquarefeet','finishedsquarefeet12',
		'taxvaluedollarcnt', 'structuretaxvaluedollarcnt','landtaxvaluedollarcnt','regionidzip', 'longitudetrim',
		'latitudetrim', 'lotsizesquarefeet','rawcensustractandblock','propertyzoningdesc','censustractandblock',
		'finishedsquarefeet15', 'regionidcity','regionidneighborhood', 'propertycountylandusecode', 
		'finishedsquarefeet6', 'garagetotalsqft', 'bedroomcnt',   
		'taxdelinquencyyear', 'finishedfloor1squarefeet','bathroomcnt', 'buildingqualitytypeid','hasaircond', 
		'roomcnt', 'garagecarcnt', 'hasheat', 'has34bath',
		'heatingorsystemtypeid',  'propertylandusetypeid',
		'taxdelinquencyflag',
		 'hasdeck', 'poolcnt',
		'finishedsquarefeet13', 'hasstories', 'fireplacecnt',
		'yardbuildingsqft17', 'basementsqft', 'numberofstories','airconditioningtypeid_meanshftenc0', 'regionidzip_meanshftenc0', 'regionidneighborhood_meanshftenc0', 
		'propertycountylandusecode_meanshftenc0', 'rawcensustractandblock_meanshftenc0','censustractandblock_meanshftenc0', 
		'regionidcity_meanshftenc0', 'latitudetrim_meanshftenc0', 'longitudetrim_meanshftenc0', 'buildingqualitytypeid_meanshftenc0',
		'bedroomcnt_meanshftenc0', 'unitcnt_meanshftenc0','calculatedbathnbr_meanshftenc0', 'propertylandusetypeid_meanshftenc0',
		'taxdelinquencyyear_meanshftenc0','propertyzoningdesc_meanshftenc0', 'taxdelinquencyflag_meanshftenc0','fullbathcnt_meanshftenc0',
		'heatingorsystemtypeid_meanshftenc0', 'roomcnt_meanshftenc0',
		'garagecarcnt_meanshftenc0', 'bathroomcnt_meanshftenc0', 'pooltypeid2_meanshftenc0','poolcnt_meanshftenc0', 'decktypeid_meanshftenc0',
		'regionidcounty_meanshftenc0', 'pooltypeid7_meanshftenc0','numberofstories_meanshftenc0', 'fireplacecnt_meanshftenc0', 
		'threequarterbathnbr_meanshftenc0', 
		'hashottuborspa_meanshftenc0', 'pooltypeid10_meanshftenc0', 'architecturalstyletypeid_meanshftenc0', 'yardbuildingsqft17_meanshftenc0',
		'fireplaceflag_meanshftenc0', 'assessmentyear_meanshftenc0','buildingclasstypeid_meanshftenc0','typeconstructiontypeid_meanshftenc0',
		'storytypeid_meanshftenc0', 'yardbuildingsqft26_meanshftenc0', 'month','qtr', 
		'X1_Mon_logerror3len',
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
'''
'X4_Mon_logerror3std',
'X3_Mon_logerror6skew',
'X3_Mon_logerror3mean',
'X4_Mon_logerror3skew',
'X3_Mon_logerror3std',
'X1_Mon_logerror6skew',
'X1_Mon_logerror3std',
'X1_Mon_logerror3skew',
'X3_Mon_logerror3skew',
'X2_Mon_logerror3skew',
'X4_Mon_logerror6skew',
'X4_Mon_logerror6std',
'X4_Mon_logerror3mean',
'X1_Mon_logerror6mean']
'''

train = joblib.load("../input/trainstat.pkl")
x = train.yrmonth
train = train[cols]
y = joblib.load("../input/y.pkl")
lbound = np.mean(y) - 3 * np.std(y) #
ubound = np.mean(y) + 3 * np.std(y) #

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

ymean = y.mean()
test = joblib.load("../input/teststat.pkl")
test = test[cols]
#test = valid # remove
oobtest = np.zeros((test.shape[0],1))
oobval = np.zeros((train.shape[0],1))
valerr = []
cnt = 0
val_scores = []
cv_r2 = []
nfold = 5
nbag =1
for i in [6]:
    for seed in [2017]:
        kf = model_selection.KFold(n_splits=nfold, shuffle=False, random_state=seed)
        for dev_index, val_index in kf.split(y): # explain for regression convert y to bins and use that for split
            dev_X, val_X = train.iloc[dev_index,:], train.iloc[val_index,:]
            dev_y, val_y = y[dev_index], y[val_index]
            dev_X = dev_X[(dev_y > lbound) & (dev_y < ubound)]
            dev_y = dev_y[(dev_y > lbound) & (dev_y < ubound)]
            val_X2 = val_X[(val_y > lbound) & (val_y < ubound)]
            val_y2 = val_y[(val_y > lbound) & (val_y < ubound)]
            
            print(dev_X.shape)  
            param = {
                'eta': 0.03,
                'objective': 'reg:linear',
                'eval_metric': 'mae',
                'base_score': dev_y.mean(),
                'silent': 1,
                'seed' : seed,    
                'subsample': .90702477560273065,
                'min_child_weight' : .5,
                'colsample_bytree' :0.95,
                'alpha' : 1.2880163765883532,
                'lambda' : 3.2778512749423294,
                'gamma' : 0.015960897098705386,
                'colsample_bylevel' : 0.95824703637227093,
                'max_depth': i
            }
            xgtrain = xgb.DMatrix(dev_X, label=dev_y)
            xgval = xgb.DMatrix(val_X2, label=val_y2)

            watchlist = [ (xgtrain,'train'), (xgval, 'val') ] 
            model = xgb.train(param, xgtrain, 1310, watchlist, verbose_eval=100)           #early_stopping_rounds=20, 
            xgval = xgb.DMatrix(val_X)
            preds = model.predict(xgval).reshape(-1,1)

            oobval[val_index,:] += preds

            cv_r2.append(mean_absolute_error(val_y, preds))    
            print(cv_r2, np.mean(cv_r2),"---", np.std(cv_r2))
            val_scores.append(mean_absolute_error(model.predict(xgb.DMatrix(valid)), yval))
            print(val_scores, np.mean(val_scores),"---", np.std(val_scores))            
            predtst = model.predict(xgb.DMatrix(test)).reshape(-1,1)
            oobtest += predtst
pred = oobtest / (nfold * nbag)
oobpred = oobval / nbag
joblib.dump(oobpred,'../input/train_xgb_meanshft_stat_enc_wo.pkl')
joblib.dump(pred,'../input/test_xgb_meanshft_stat_enc_wo.pkl')

############################################
# XGB diff column set
############################################
###XGB Finale - No Outliers
from sklearn import model_selection
import xgboost as xgb
import pandas as pd
import numpy as np
import datetime as dt
import gc
from sklearn.metrics import mean_absolute_error
from sklearn.externals import joblib
##########################################################
# Mean Encode
##########################################################
cols = ['resell', 'latitude', 'countnullcol', 'yearbuilt', 'longitude',
		'taxamount','calculatedfinishedsquarefeet','finishedsquarefeet12',
		'taxvaluedollarcnt', 'structuretaxvaluedollarcnt','landtaxvaluedollarcnt','regionidzip', 'longitudetrim',
		'latitudetrim', 'lotsizesquarefeet','rawcensustractandblock','propertyzoningdesc','censustractandblock',
		'finishedsquarefeet15', 'regionidcity','regionidneighborhood', 'propertycountylandusecode', 
		'finishedsquarefeet6', 'garagetotalsqft', 'bedroomcnt',   
		'taxdelinquencyyear', 'finishedfloor1squarefeet','bathroomcnt', 'buildingqualitytypeid','hasaircond', 
		'roomcnt', 'garagecarcnt', 'hasheat', 'has34bath',
		'heatingorsystemtypeid',  'propertylandusetypeid',
		'taxdelinquencyflag',
		 'hasdeck', 'poolcnt',
		'finishedsquarefeet13', 'hasstories', 'fireplacecnt',
		'yardbuildingsqft17', 'basementsqft', 'numberofstories','airconditioningtypeid_meanshftenc0', 'regionidzip_meanshftenc0', 'regionidneighborhood_meanshftenc0', 
		'propertycountylandusecode_meanshftenc0', 'rawcensustractandblock_meanshftenc0','censustractandblock_meanshftenc0', 
		'regionidcity_meanshftenc0', 'latitudetrim_meanshftenc0', 'longitudetrim_meanshftenc0', 'buildingqualitytypeid_meanshftenc0',
		'bedroomcnt_meanshftenc0', 'unitcnt_meanshftenc0','calculatedbathnbr_meanshftenc0', 'propertylandusetypeid_meanshftenc0',
		'taxdelinquencyyear_meanshftenc0','propertyzoningdesc_meanshftenc0', 'taxdelinquencyflag_meanshftenc0','fullbathcnt_meanshftenc0',
		'heatingorsystemtypeid_meanshftenc0', 'roomcnt_meanshftenc0',
		'garagecarcnt_meanshftenc0', 'bathroomcnt_meanshftenc0', 'pooltypeid2_meanshftenc0','poolcnt_meanshftenc0', 'decktypeid_meanshftenc0',
		'regionidcounty_meanshftenc0', 'pooltypeid7_meanshftenc0','numberofstories_meanshftenc0', 'fireplacecnt_meanshftenc0', 
		'threequarterbathnbr_meanshftenc0', 
		'hashottuborspa_meanshftenc0', 'pooltypeid10_meanshftenc0', 'architecturalstyletypeid_meanshftenc0', 'yardbuildingsqft17_meanshftenc0',
		'fireplaceflag_meanshftenc0', 'assessmentyear_meanshftenc0','buildingclasstypeid_meanshftenc0','typeconstructiontypeid_meanshftenc0',
		'storytypeid_meanshftenc0', 'yardbuildingsqft26_meanshftenc0', 'month','qtr']

train = joblib.load("../input/trainstat.pkl")
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
#test = valid # remove
oobtest = np.zeros((test.shape[0],1))
oobval = np.zeros((train.shape[0],1))
valerr = []
cnt = 0
val_scores = []
cv_r2 = []
nfold = 5
nbag =1
gc.collect()
for i in [7]:
    for seed in [2017]:
        kf = model_selection.KFold(n_splits=nfold, shuffle=False, random_state=seed)
        for dev_index, val_index in kf.split(y): # explain for regression convert y to bins and use that for split
            dev_X, val_X = train.iloc[dev_index,:], train.iloc[val_index,:]
            dev_y, val_y = y[dev_index], y[val_index]
            dev_X = dev_X[(dev_y > lbound) & (dev_y < ubound)]
            dev_y = dev_y[(dev_y > lbound) & (dev_y < ubound)]
            val_X2 = val_X[(val_y > lbound) & (val_y < ubound)]
            val_y2 = val_y[(val_y > lbound) & (val_y < ubound)]
            print(dev_X.shape)  
            param = {
                'eta': 0.03,
                'objective': 'reg:linear',
                'eval_metric': 'mae',
                'base_score': dev_y.mean(),
                'silent': 1,
                'seed' : seed,    
                'subsample': .90702477560273065,
                'min_child_weight' : 4.,
                'colsample_bytree' :0.95,
                'alpha' : 1.2880163765883532,
                'lambda' : 3.2778512749423294,
                'gamma' : 0.015960897098705386,
                'colsample_bylevel' : 0.95824703637227093,
                'max_depth': i
            }
            xgtrain = xgb.DMatrix(dev_X, label=dev_y)
            xgval = xgb.DMatrix(val_X2, label=val_y2)

            watchlist = [ (xgtrain,'train'), (xgval, 'val') ] 
            model = xgb.train(param, xgtrain, 1201, watchlist, verbose_eval=100)           #early_stopping_rounds=20, 
            xgval = xgb.DMatrix(val_X)
            preds = model.predict(xgval).reshape(-1,1)

            oobval[val_index,:] += preds

            cv_r2.append(mean_absolute_error(val_y, preds))    
            print(cv_r2, np.mean(cv_r2),"---", np.std(cv_r2))
            val_scores.append(mean_absolute_error(model.predict(xgb.DMatrix(valid)), yval))
            print(val_scores, np.mean(val_scores),"---", np.std(val_scores))            
            predtst = model.predict(xgb.DMatrix(test)).reshape(-1,1)
            oobtest += predtst
pred = oobtest / (nfold * nbag)
oobpred = oobval / nbag
joblib.dump(oobpred,'../input/train_xgb_meanshft_enc_wo.pkl')
joblib.dump(pred,'../input/test_xgb_meanshft_enc_wo.pkl')


##################Linear
ymean = y.mean()

test = test[cols]
oobtest = np.zeros((test.shape[0],1))
oobval = np.zeros((train.shape[0],1))
valerr = []
cnt = 0
val_scores = []
cv_r2 = []
nfold = 5
nbag =1
for seed in [2017,]:
    kf = model_selection.KFold(n_splits=nfold, shuffle=False, random_state=seed)
    for dev_index, val_index in kf.split(y): # explain for regression convert y to bins and use that for split
        dev_X, val_X = train.iloc[dev_index,:], train.iloc[val_index,:]
        dev_y, val_y = y[dev_index], y[val_index]
        dev_X = dev_X[(dev_y > lbound) & (dev_y < ubound)]
        dev_y = dev_y[(dev_y > lbound) & (dev_y < ubound)]
        val_X2 = val_X[(val_y > lbound) & (val_y < ubound)]
        val_y2 = val_y[(val_y > lbound) & (val_y < ubound)]
        print(dev_X.shape)  
        
        param = {
            'booster':'gblinear',
            'eta': 0.03,
            'objective': 'reg:linear',
            'eval_metric': 'mae',
            'base_score': dev_y.mean(),
            'silent': 1,
            'seed' : seed, 
            'nthread': 1,   
            'subsample': .9,
            'min_child_weight' : 4.0,
            'colsample_bytree' :0.95,
            'alpha' : 10.,
            'lambda' : 1.
        }
        xgtrain = xgb.DMatrix(dev_X, label=dev_y)
        xgval = xgb.DMatrix(val_X2, label=val_y2)

        watchlist = [ (xgtrain,'train'), (xgval, 'val') ] 
        model = xgb.train(param, xgtrain, 1200, watchlist, verbose_eval=100)           
        xgval = xgb.DMatrix(val_X)
        preds = model.predict(xgval).reshape(-1,1)

        oobval[val_index,:] += preds

        cv_r2.append(mean_absolute_error(val_y, preds))    
        print(cv_r2, np.mean(cv_r2),"---", np.std(cv_r2))
        val_scores.append(mean_absolute_error(model.predict(xgb.DMatrix(valid)), yval))
        print(val_scores, np.mean(val_scores),"---", np.std(val_scores))            
        xgval = xgb.DMatrix(test)        
        predtst = model.predict(xgval).reshape(-1,1)
        oobtest += predtst
pred1 = oobtest / (nfold * nbag)
oobpred1 = oobval / nbag
joblib.dump(oobpred1,'../input/train_xgb_lr_wo.pkl')
joblib.dump(pred1,'../input/test_xgb_lr_wo.pkl')

