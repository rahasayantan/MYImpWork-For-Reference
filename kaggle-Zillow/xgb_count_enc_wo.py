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
       'yardbuildingsqft17', 'basementsqft', 'numberofstories',
       'taxdelinquencyyear_countenc0', 
       'regionidzip_countenc0', 
       'latitudetrim_countenc0',
        'regionidcity_countenc0',
       'longitudetrim_countenc0', 'propertycountylandusecode_countenc0',
       'fullbathcnt_countenc0', 'regionidneighborhood_countenc0',
       'bedroomcnt_countenc0',
       'censustractandblock_countenc0',
       'architecturalstyletypeid_countenc0', 'unitcnt_countenc0',
       'poolcnt_countenc0',
       'rawcensustractandblock_countenc0', 
       'buildingqualitytypeid_countenc0',
       'propertylandusetypeid_countenc0', 'yardbuildingsqft17_countenc0',
       'airconditioningtypeid_countenc0',
       'calculatedbathnbr_countenc0',
       'yardbuildingsqft26_countenc0', 
       'decktypeid_countenc0', 'pooltypeid7_countenc0',
       'roomcnt_countenc0', 
       'heatingorsystemtypeid_countenc0', 
	   'propertyzoningdesc_countenc0',
       'storytypeid_countenc0',
       'fireplacecnt_countenc0', 'threequarterbathnbr_countenc0',
       'pooltypeid10_countenc0', 'pooltypeid2_countenc0',
       'hashottuborspa_countenc0',
       'regionidcounty_countenc0',
       'fireplaceflag_countenc0',
       'typeconstructiontypeid_countenc0', 'numberofstories_countenc0',
       'buildingclasstypeid_countenc0', 'garagecarcnt_countenc0',
       'taxdelinquencyflag_countenc0',
       'assessmentyear_countenc0', 
       'bathroomcnt_countenc0', 'month', 'qtr']


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
                'eta': 0.01,
                'objective': 'reg:linear',
                'eval_metric': 'mae',
                'base_score': dev_y.mean(),
                'silent': 1,
                'seed' : seed,    
                'subsample': .98,
                'min_child_weight' : 1.03,
                'colsample_bytree' :0.9,
                'alpha' : 1.0,
                'lambda' : 1.0,
                'gamma' : 0.02,
                'colsample_bylevel' : 0.9,
                'max_depth': i  
            }
            xgtrain = xgb.DMatrix(dev_X, label=dev_y)
            xgval = xgb.DMatrix(val_X2, label=val_y2)

            watchlist = [ (xgtrain,'train'), (xgval, 'val') ] 
            model = xgb.train(param, xgtrain, 500, watchlist, verbose_eval=100)           #early_stopping_rounds=20, 
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
joblib.dump(oobpred,'../input/train_xgb_count_enc_wo.pkl')
joblib.dump(pred,'../input/test_xgb_count_enc_wo.pkl')

