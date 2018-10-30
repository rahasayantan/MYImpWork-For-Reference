from sklearn import neighbors
###XGB Finale - No Outliers
from sklearn import model_selection, preprocessing, ensemble
import pandas as pd
import numpy as np
import datetime as dt
import gc
from sklearn.metrics import mean_absolute_error
from sklearn.externals import joblib

train = joblib.load("../input/trainstat.pkl")

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
		'storytypeid_meanshftenc0', 'yardbuildingsqft26_meanshftenc0', 'month','qtr']

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

from sklearn.preprocessing import MinMaxScaler
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
nfold = 5

n_neighbors = 64
np.random.seed(2017)
seed = 2017
gc.collect()
for i, weights in enumerate(['uniform', 'distance']):
    for n_neighbors in [2, 8, 32, 64]:
        kf = model_selection.KFold(n_splits=nfold, shuffle=True, random_state=seed)    
        for dev_index, val_index in kf.split(train):
            dev_X, val_X = train.values[dev_index], train.values[val_index]
            dev_y, val_y = y[dev_index], y[val_index]
            dev_X = dev_X[(dev_y >= lbound) & (dev_y <= ubound)]
            dev_y = dev_y[(dev_y >= lbound) & (dev_y <= ubound)]
            print(dev_X.shape)  

            et = neighbors.KNeighborsRegressor(n_neighbors, weights=weights)
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
pred2 = oobtest / (8 * nfold)
oobpred2 = oobval / 8
print(mean_absolute_error(y, oobpred2))

######################################################
# Save for L2 Predictors
######################################################
joblib.dump(oobpred2,'../input/train_knn_withoutlier.pkl')
joblib.dump(pred2,'../input/test_knn_withoutlier.pkl')

