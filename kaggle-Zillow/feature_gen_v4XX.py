###########################################################
# Encode
###########################################################

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn import preprocessing,model_selection, ensemble
from sklearn.preprocessing import LabelEncoder
import scipy.stats as ss
from sklearn.externals import joblib
from scipy.sparse import csr_matrix

def cat2MedianShiftEncode(train_char, test_char, y, nbag = 10, nfold = 20, minCount = 3):
    train_df = train_char.copy()
    test_df = test_char.copy()
    for c in train_char.columns:
        data = train_char[[c]].copy()
        data['y'] = y
        med_y = np.median(y)
        enc_mat = np.zeros((y.shape[0],1))
        enc_mat_test = np.zeros((test_char.shape[0],1))
        for bag in np.arange(nbag):
        
            kf = model_selection.KFold(n_splits= nfold, shuffle=True, random_state=2017*bag)
            for dev_index, val_index in kf.split(range(data['y'].shape[0])):
                dev_X, val_X = data.iloc[dev_index,:], data.iloc[val_index,:]
                datax = dev_X.groupby(c).agg([len,np.mean,np.std, np.median])
                datax.columns = ['_'.join(col).strip() for col in datax.columns.values]
                datax = datax.loc[datax.y_len > minCount]
                datax[c+'_medshftenc'] =  datax['y_median']-med_y
                datax.drop(['y_len','y_mean','y_std','y_median'],axis=1,inplace=True)
                print(datax.columns)
                datatst = test_char[[c]].copy()
                val_X = val_X.join(datax,on=[c], how='left').fillna(0)
                datatst = datatst.join(datax,on=[c], how='left').fillna(0)
                enc_mat[val_index,...] += val_X[list(set(datax.columns)-set([c]))]
                enc_mat_test += datatst[list(set(datax.columns)-set([c]))]

        enc_mat_test /= (nfold * nbag)
        enc_mat /= (nbag)        
        enc_mat = pd.DataFrame(enc_mat)  
        enc_mat.columns=[c+'_medshftenc'+str(x) for x in enc_mat.columns] 
        enc_mat_test = pd.DataFrame(enc_mat_test)  
        enc_mat_test.columns=[enc_mat.columns] 
        train_df = pd.concat((enc_mat.reset_index(drop = True),train_df.reset_index(drop = True)), axis=1)
        test_df = pd.concat([enc_mat_test.reset_index(drop = True),test_df.reset_index(drop = True)],axis=1)
    return train_df, test_df

def cat2MeanShiftEncode(train_char, test_char, y, nbag = 10, nfold = 20, minCount = 3):
    train_df = train_char.copy()
    test_df = test_char.copy()
    for c in train_char.columns:
        data = train_char[[c]].copy()
        data['y'] = y
        mean_y = np.mean(y)
        enc_mat = np.zeros((y.shape[0],1))
        enc_mat_test = np.zeros((test_char.shape[0],1))
        for bag in np.arange(nbag):
            
            kf = model_selection.KFold(n_splits= nfold, shuffle=True, random_state=2017*bag)
            for dev_index, val_index in kf.split(range(data['y'].shape[0])):
                dev_X, val_X = data.iloc[dev_index,:], data.iloc[val_index,:]
                datax = dev_X.groupby(c).agg([len,np.mean,np.std, np.median])
                datax.columns = ['_'.join(col).strip() for col in datax.columns.values]
                datax = datax.loc[datax.y_len > minCount]
                datax[c+'_meanshftenc'] =  datax['y_mean'] - mean_y
                datax.drop(['y_len','y_mean','y_std','y_median'],axis=1,inplace=True)
                print(datax.columns)
                datatst = test_char[[c]].copy()
                val_X = val_X.join(datax,on=[c], how='left').fillna(0)
                datatst = datatst.join(datax,on=[c], how='left').fillna(0)
                enc_mat[val_index,...] += val_X[list(set(datax.columns)-set([c]))]
                enc_mat_test += datatst[list(set(datax.columns)-set([c]))]

        enc_mat_test /= (nfold*nbag)
        enc_mat /= (nbag)        
        enc_mat = pd.DataFrame(enc_mat)  
        enc_mat.columns=[c+'_meanshftenc'+str(x) for x in enc_mat.columns] 
        enc_mat_test = pd.DataFrame(enc_mat_test)  
        enc_mat_test.columns=[enc_mat.columns] 
        train_df = pd.concat((enc_mat.reset_index(drop = True),train_df.reset_index(drop = True)), axis=1)
        test_df = pd.concat([enc_mat_test.reset_index(drop = True),test_df.reset_index(drop = True)],axis=1)
    return train_df, test_df

def cat2MeanEncode(train_char, test_char, y, nbag = 10, nfold = 20, minCount = 3):
    train_df = train_char.copy()
    test_df = test_char.copy()
    rn = np.mean(y)
    for c in train_char.columns:
        data = train_char[[c]].copy()
        data['y'] = y
        enc_mat = np.zeros((y.shape[0],1))
        enc_mat_test = np.zeros((test_char.shape[0],1))
        for bag in np.arange(nbag):
                
            kf = model_selection.KFold(n_splits= nfold, shuffle=True, random_state=2017*bag)
            for dev_index, val_index in kf.split(range(data['y'].shape[0])):
                dev_X, val_X = data.iloc[dev_index,:], data.iloc[val_index,:]
                datax = dev_X.groupby(c).agg([len,np.mean,np.std, np.median])
                datax.columns = ['_'.join(col).strip() for col in datax.columns.values]
                datax = datax.loc[datax.y_len > minCount]
                datax[c+'_meanenc'] =  datax['y_mean']
                datax.drop(['y_len','y_mean','y_std','y_median'],axis=1,inplace=True)
                print(datax.columns)
                datatst = test_char[[c]].copy()
                val_X = val_X.join(datax,on=[c], how='left').fillna(rn)
                datatst = datatst.join(datax,on=[c], how='left').fillna(rn)
                enc_mat[val_index,...] += val_X[list(set(datax.columns)-set([c]))]
                enc_mat_test += datatst[list(set(datax.columns)-set([c]))]

        enc_mat_test /= (nfold*nbag)
        enc_mat /= (nbag)        
        enc_mat = pd.DataFrame(enc_mat)  
        enc_mat.columns=[c+'_meanenc'+str(x) for x in enc_mat.columns] 
        enc_mat_test = pd.DataFrame(enc_mat_test)  
        enc_mat_test.columns=[enc_mat.columns] 
        train_df = pd.concat((enc_mat,train_df), axis=1)
        test_df = pd.concat([enc_mat_test,test_df],axis=1)
    return train_df, test_df

def cat2MedianEncode(train_char, test_char, y, nbag = 10, nfold = 20, minCount = 3):
    train_df = train_char.copy()
    test_df = test_char.copy()
    rn = np.mean(y)
    for c in train_char.columns:
        data = train_char[[c]].copy()
        data['y'] = y
        enc_mat = np.zeros((y.shape[0],1))
        enc_mat_test = np.zeros((test_char.shape[0],1))
        for bag in np.arange(nbag):

            kf = model_selection.KFold(n_splits= nfold, shuffle=True, random_state=2017*bag)
            for dev_index, val_index in kf.split(range(data['y'].shape[0])):
                dev_X, val_X = data.iloc[dev_index,:], data.iloc[val_index,:]
                datax = dev_X.groupby(c).agg([len,np.mean,np.std, np.median])
                datax.columns = ['_'.join(col).strip() for col in datax.columns.values]
                datax = datax.loc[datax.y_len > minCount]
                datax[c+'_medianenc'] =  datax['y_mean']
                datax.drop(['y_len','y_mean','y_std','y_median'],axis=1,inplace=True)
                print(datax.columns)
                datatst = test_char[[c]].copy()
                val_X = val_X.join(datax,on=[c], how='left').fillna(rn)
                datatst = datatst.join(datax,on=[c], how='left').fillna(rn)
                enc_mat[val_index,...] += val_X[list(set(datax.columns)-set([c]))]
                enc_mat_test += datatst[list(set(datax.columns)-set([c]))]

        enc_mat_test /= (nfold*nbag)
        enc_mat /= (nbag)        
        enc_mat = pd.DataFrame(enc_mat)  
        enc_mat.columns=[c+'_medianenc'+str(x) for x in enc_mat.columns] 
        enc_mat_test = pd.DataFrame(enc_mat_test)  
        enc_mat_test.columns=[enc_mat.columns] 
        train_df = pd.concat((enc_mat,train_df), axis=1)
        test_df = pd.concat([enc_mat_test,test_df],axis=1)
    return train_df, test_df

def countEncode(train_char, test_char, y, nbag = 10, nfold = 20, minCount = 3):
    train_df = train_char.copy()
    test_df = test_char.copy()
    rn = 999
    for c in train_char.columns:
        data = train_char[[c]].copy()
        data['y'] = y
        enc_mat = np.zeros((y.shape[0],1))
        enc_mat_test = np.zeros((test_char.shape[0],1))
        for bag in np.arange(nbag):
            
            kf = model_selection.KFold(n_splits= nfold, shuffle=True, random_state=2017*bag)
            for dev_index, val_index in kf.split(range(data['y'].shape[0])):
                dev_X, val_X = data.iloc[dev_index,:], data.iloc[val_index,:]
                datax = dev_X.groupby(c).agg([len,np.mean,np.std, np.median])
                datax.columns = ['_'.join(col).strip() for col in datax.columns.values]
                datax = datax.loc[datax.y_len > minCount]
                datax[c+'_countenc'] =  datax['y_len']
                datax.drop(['y_len','y_mean','y_std','y_median'],axis=1,inplace=True)
                print(datax.columns)
                datatst = test_char[[c]].copy()
                val_X = val_X.join(datax,on=[c], how='left').fillna(rn)
                datatst = datatst.join(datax,on=[c], how='left').fillna(rn)
                enc_mat[val_index,...] += val_X[list(set(datax.columns)-set([c]))]
                enc_mat_test += datatst[list(set(datax.columns)-set([c]))]

        enc_mat_test /= (nfold * nbag)
        enc_mat /= nbag
        enc_mat = pd.DataFrame(enc_mat)  
        enc_mat.columns=[c+'_countenc'+str(x) for x in enc_mat.columns] 
        enc_mat_test = pd.DataFrame(enc_mat_test)  
        enc_mat_test.columns=[enc_mat.columns] 
        train_df = pd.concat((enc_mat,train_df), axis=1)
        test_df = pd.concat([enc_mat_test,test_df],axis=1)
    return train_df, test_df

def rankCountEncode(train_char, test_char, y, nbag = 10, nfold = 20, minCount = 3):
    train_df = train_char.copy()
    test_df = test_char.copy()
    rn = 999
    for c in train_char.columns:
        data = train_char[[c]].copy()
        data['y'] = y
        enc_mat = np.zeros((y.shape[0],1))
        enc_mat_test = np.zeros((test_char.shape[0],1))
    
        for bag in np.arange(nbag):
            kf = model_selection.KFold(n_splits= nfold, shuffle=True, random_state=2017*bag)
            for dev_index, val_index in kf.split(range(data['y'].shape[0])):
                dev_X, val_X = data.iloc[dev_index,:], data.iloc[val_index,:]
                datax = dev_X.groupby(c).agg([len,np.mean,np.std, np.median])
                datax.columns = ['_'.join(col).strip() for col in datax.columns.values]
                datax = datax.loc[datax.y_len > minCount]
                datax[c+'_rankenc'] =  datax['y_len']
                datax[c+'_rankenc'] =  ss.rankdata(datax[c+'_rankenc'].values)
                datax.drop(['y_len','y_mean','y_std','y_median'],axis=1,inplace=True)
                print(datax.columns)
                datatst = test_char[[c]].copy()
                val_X = val_X.join(datax,on=[c], how='left').fillna(rn)
                datatst = datatst.join(datax,on=[c], how='left').fillna(rn)
                enc_mat[val_index,...] += val_X[list(set(datax.columns)-set([c]))]
                enc_mat_test += datatst[list(set(datax.columns)-set([c]))]

        enc_mat_test /= (nfold * nbag)
        enc_mat /= (nbag)
        enc_mat = pd.DataFrame(enc_mat)  
        enc_mat.columns=[c+'_rankenc'+str(x) for x in enc_mat.columns] 
        enc_mat_test = pd.DataFrame(enc_mat_test)  
        enc_mat_test.columns=[enc_mat.columns] 
        train_df = pd.concat((enc_mat,train_df), axis=1)
        test_df = pd.concat([enc_mat_test,test_df],axis=1)
    return train_df, test_df

def catLabelEncode(train_char, test_char):
    train_df = train_char.copy()
    test_df = test_char.copy()
    train_test = pd.concat((train_df,test_df))
    for feat in train_df.columns:
        train_test[feat] = pd.factorize(train_test[feat])[0]
    train_df = train_test.iloc[:train_char.shape[0],:]
    test_df = train_test.iloc[train_char.shape[0]:,:]    
    return train_df, test_df

def OHCEncode(train_char, test_char):
    train_df = train_char.copy()
    test_df = test_char.copy()
    train_test = pd.concat((train_df,test_df))
    ohe = csr_matrix(pd.get_dummies(train_test, dummy_na=False, sparse=True))    
    train_df = ohe[:train_df.shape[0],:]
    test_df = ohe[train_df.shape[0]:,:]    
    return train_df, test_df

###########################################################
# START
###########################################################

import pandas as pd
import numpy as np
import datetime as dt
import gc
from sklearn.preprocessing import MinMaxScaler
from sklearn.externals import joblib
#from low_memory import reduce_mem_usage

print('Loading train data ...')

train1 = pd.read_csv('../input/train_2016_v2.csv')
train2 = pd.read_csv('../input/train_2017.csv')

#prop1 = pd.read_csv('../input/properties_2016.csv')
prop2 = pd.read_csv('../input/properties_2017.csv')


print('Binding to float32')

for c, dtype in zip(prop2.columns, prop2.dtypes):
	if dtype == np.float64:
#		prop1[c] = prop1[c].astype(np.float32)
		prop2[c] = prop2[c].astype(np.float32)
		
train1 = train1.merge(prop2, how='left', on='parcelid') # change this to prop2
train2 = train2.merge(prop2, how='left', on='parcelid')

dftrain = pd.concat((train1, train2))
del(train1, train2); gc.collect()

trainx = dftrain[['parcelid', 'transactiondate']].groupby(['parcelid','transactiondate']).agg('size').groupby(level=[0]).cumsum()
trainx = trainx.reset_index()
trainx = trainx[trainx[0] > 1]
dftrain['resell'] = 0

keys = ['parcelid', 'transactiondate']
i1 = dftrain.set_index(keys).index
i2 = trainx.set_index(keys).index
dftrain.loc[i1.isin(i2), 'resell'] = 1
dftrain.reset_index(drop = True, inplace = True)
del(trainx, i2); gc.collect()
###########################################
sample = pd.read_csv('../input/sample_submission.csv')
test = sample[['ParcelId']]
test.columns = ['parcelid']
dftest = test.merge(prop2, how='left', on='parcelid')# change this to prop2
dftest['transactiondate'] = '2017-10-01'
dftest['logerror'] = 0
dftest['resell'] = 0
keys = ['parcelid']
i1 = dftrain.set_index(keys).index
i2 = dftest.set_index(keys).index
dftest.loc[i2.isin(i1), 'resell'] = 1

del(i1, i2, test, sample, prop2)#prop1, 
gc.collect()

def featureGen(x):
    x['countnullcol'] = x.isnull().sum(axis = 1)
    x["transactiondate"] = pd.to_datetime(x["transactiondate"])
    x["yr"] = pd.DatetimeIndex(x["transactiondate"]).year  
    x["month"] = pd.DatetimeIndex(x["transactiondate"]).month
    x["qtr"] = pd.DatetimeIndex(x["transactiondate"]).quarter

    x['latitudetrim'] = x['latitude'].fillna(999999999)/10000
    x['latitudetrim'] = x['latitudetrim'].astype(int)
    x['longitudetrim'] = x['longitude'].fillna(999999999)/10000
    x['longitudetrim'] = x['longitudetrim'].astype(int)
    x['yearbuilt'] = x['yearbuilt'].fillna(1700)
    x['yearbuilt'] = 2017 - x.yearbuilt
    ## Binary columns for features -> Zero means its has it and 1 means feature is absent
    x['hasaircond'] = np.isnan(x.airconditioningtypeid).astype(int) 
    x['hasdeck'] = np.isnan(x.decktypeid).astype(int) 
    x['has34bath'] = np.isnan(x.threequarterbathnbr).astype(int)
    x['hasfullbath'] = np.isnan(x.fullbathcnt).astype(int)
    x['hashottuborspa'] = x.hashottuborspa.fillna(False).astype(int)
    x['hasheat'] = np.isnan(x.heatingorsystemtypeid).astype(int)
    x['hasstories'] = np.isnan(x.numberofstories).astype(int)
    x['haspatio'] = np.isnan(x.yardbuildingsqft17).astype(int)
    x['taxdelinquencyyear'] = 2017 - x['taxdelinquencyyear']
    return x
    
#['regionidzip_meanshftenc0', 'buildingclasstypeid_meanshftenc0', 'propertycountylandusecode_meanshftenc0', 'propertyzoningdesc_meanshftenc0', 'rawcensustractandblock_meanshftenc0', 'taxdelinquencyflag_meanshftenc0', 'countnullcol', 'regionidneighborhood_meanshftenc0']

print("Start ftr gen..")
    
dftest = featureGen(dftest)
#joblib.dump(dftest,"../input/testNewFtr1.pkl")

dftrain = featureGen(dftrain)
#joblib.dump(dftrain,"../input/trainNewFtr1.pkl")

catcols = ['bathroomcnt', 'bedroomcnt', 'buildingqualitytypeid', 'buildingclasstypeid', 'calculatedbathnbr', 'decktypeid', 'threequarterbathnbr', 'fips', 'fireplacecnt', 'fireplaceflag', 'fullbathcnt', 'garagecarcnt', 'hashottuborspa', 'heatingorsystemtypeid', 'numberofstories', 'poolcnt', 'pooltypeid10', 'pooltypeid2', 'pooltypeid7', 'propertycountylandusecode', 'propertylandusetypeid', 'propertyzoningdesc', 'rawcensustractandblock', 'censustractandblock', 'regionidcounty', 'regionidcity', 'regionidzip', 'regionidneighborhood','storytypeid', 'typeconstructiontypeid', 'unitcnt', 'yardbuildingsqft17', 'yardbuildingsqft26', 'assessmentyear','taxdelinquencyflag','taxdelinquencyyear', 'roomcnt', 'latitudetrim', 'longitudetrim','airconditioningtypeid','architecturalstyletypeid']

numcols = ['countnullcol', 'latitude', 'longitude', 'yearbuilt', 'hasaircond', 'hasdeck', 'has34bath', 'hasfullbath', 'hasheat', 'hasstories', 'haspatio', 'basementsqft', 'finishedfloor1squarefeet', 'calculatedfinishedsquarefeet', 'finishedsquarefeet6', 'finishedsquarefeet12', 'finishedsquarefeet13', 'finishedsquarefeet15', 'finishedsquarefeet50', 'lotsizesquarefeet', 'garagetotalsqft', 'poolsizesum','taxvaluedollarcnt', 'structuretaxvaluedollarcnt', 'landtaxvaluedollarcnt', 'taxamount', 'logerror','parcelid', 'yr','month','qtr', 'resell']# ,'transactiondate'

### numerical processing
print("Num processing..")
dftrainnum = dftrain[numcols].fillna(-9)
dftestnum = dftest[numcols].fillna(-9)

############################
dftraincat = dftrain[catcols].fillna(-9)
dftestcat = dftest[catcols].fillna(-9)

ntrain = dftrain.shape[0]
print("Cat Processing..")
#from encoding import cat2MeanShiftEncode, catLabelEncode, countEncode, cat2MedianEncode, cat2MeanEncode, cat2MedianShiftEncode
train, test = cat2MeanShiftEncode(dftraincat.copy(), dftestcat.copy(), dftrainnum.logerror.values, nbag = 10, nfold = 20, minCount = 10)
trainmn = train.iloc[:,:41].copy()
testmn = test.iloc[:,:41].copy()

joblib.dump(trainmn,"../input/train_catMeanshftenc_v2.pkl")
joblib.dump(testmn,"../input/testcat_catMeanshftenc_v2.pkl")

#train, test = cat2MedianEncode(dftraincat.copy(), dftestcat.copy(), y, nbag = 10, nfold = 20, minCount = 10)
#joblib.dump(train.iloc[:,:41],"../input/train_catMedianenc.pkl")
#joblib.dump(test.iloc[:,:41],"../input/testcat_catMedianenc.pkl")

#train, test = cat2MeanEncode(dftraincat.copy(), dftestcat.copy(), y, nbag = 10, nfold = 20, minCount = 10)
#joblib.dump(train.iloc[:,:41],"../input/train_catMeanenc.pkl")
#joblib.dump(test.iloc[:,:41],"../input/testcat_catMeanenc.pkl")

train, test = catLabelEncode(dftraincat.copy(), dftestcat.copy())
trainlbl = train.iloc[:,:41].copy()
testlbl = test.iloc[:,:41].copy()

joblib.dump(trainlbl,"../input/trainlblcat_v2.pkl")
joblib.dump(testlbl,"../input/testcatlblcat_v2.pkl")

train, test = countEncode(dftraincat.copy(), dftestcat.copy(), dftrainnum.logerror.values, nbag = 1, nfold = 20, minCount = 10)
traincnt = train.iloc[:,:41].copy()
testcnt = test.iloc[:,:41].copy()

joblib.dump(traincnt,"../input/train_countenc_v2.pkl")
joblib.dump(testcnt,"../input/testcat_countenc_v2.pkl")

del(train, test)
gc.collect()

#joblib.dump(dftrainnum,"../input/trainnum9Impute.pkl")
#joblib.dump(dftestnum,"../input/testnum9Impute.pkl")
#traincnt = traincnt.iloc[:,:41]
#trainlbl = trainlbl.iloc[:,:41]
#trainmn = trainmn.iloc[:,:41]
##########
# Big DS
train = pd.concat((dftrainnum, traincnt, trainmn, trainlbl), axis =1)
test = pd.concat((dftestnum, testcnt, testmn, testlbl), axis =1)
del(dftrainnum, traincnt, trainmn, trainlbl); gc.collect()
del(dftestnum, testcnt, testmn, testlbl); gc.collect()


joblib.dump(train,"../input/trainmod.pkl")
joblib.dump(test,"../input/testmod.pkl")

########################
# Get stat Ftrs
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn import preprocessing,model_selection, ensemble
from sklearn.preprocessing import LabelEncoder
import scipy.stats as ss
from sklearn.externals import joblib
from scipy.sparse import csr_matrix

train = joblib.load("../input/trainmod.pkl")
test = joblib.load("../input/testmod.pkl")

print("Statistical Ftr Gen..")

train['yrmonth'] = train.apply(lambda x: str(int(x.yr)) + '-' + str(int(x.month)), axis = 1)
test['yrmonth'] = test.apply(lambda x: str(int(x.yr)) + '-' + str(int(x.month)), axis = 1)

def getStatsFtr(dftrain, dftest, skipmon =0, yrmonNum='2016-09', aggPer = 6, colLst=[], ind = '1'):
    start = pd.Period(yrmonNum) - skipmon - aggPer 
    end = pd.Period(yrmonNum) - skipmon
    dftrain['yrmonth'] = dftrain.yrmonth.apply(lambda x: pd.Period(x))
    data = dftrain[(dftrain.yrmonth >= start) & (dftrain.yrmonth < end)]
    dftest['yrmonth'] = dftest.yrmonth.apply(lambda x: pd.Period(x))
    x = dftest[(dftest["yrmonth"] == pd.Period(yrmonNum))].copy()
    data = data[colLst+['logerror']].copy()
    datax = data.groupby(colLst).agg([len,np.mean,np.std, 'skew'])
    datax.columns = ['X'+ ind +'_Mon'+'_'+str(aggPer).join(col).strip() for col in datax.columns.values]
    datax = datax.loc[datax['X' + ind + '_Mon'+'_logerror'+str(aggPer)+'len'] > 3]
    datax.reset_index(inplace = True)
    x = x.merge(datax, on = colLst, how = 'left')
    #col2send = [a for a in x.columns if 'X1_' in a]
    return x#[['parcelid','logerror']+col2send]

#dftraincat['y'] = y
#dftraincat['parcelid'] = trainparcel

def genStat(dftraincat, test, skipmon =3, aggPer = 3, collst=[], ind = '1'):
    it = 0
    for i in list(np.unique(test.yrmonth)):
        print(i)
        if it == 0:
            it += 1                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 
            x = getStatsFtr(dftraincat, test, skipmon =skipmon, yrmonNum=i, aggPer = aggPer, colLst = collst, ind = ind)
        else:
            x = pd.concat((x,getStatsFtr(dftraincat, test, skipmon =skipmon, yrmonNum=i, aggPer = aggPer, colLst = collst, ind = ind)))
    return x.fillna(-9)

colLst = ['bedroomcnt','censustractandblock']

dftrainStat = genStat(train, train, 1, 3, colLst)
dftrainStat = genStat(dftrainStat, dftrainStat, 1, 6, colLst)
dftrainStat = genStat(dftrainStat, dftrainStat, 1, 12, colLst)
dftrainStat = genStat(dftrainStat, dftrainStat, 1, 18, colLst)

colLst = ['bedroomcnt','regionidneighborhood']

dftrainStat = genStat(dftrainStat, dftrainStat, 1, 3, colLst, '2')
dftrainStat = genStat(dftrainStat, dftrainStat, 1, 6, colLst, '2')
dftrainStat = genStat(dftrainStat, dftrainStat, 1, 12, colLst, '2')
dftrainStat = genStat(dftrainStat, dftrainStat, 1, 18, colLst, '2')

colLst = ['bedroomcnt','regionidzip']

dftrainStat = genStat(dftrainStat, dftrainStat, 1, 3, colLst, '3')
dftrainStat = genStat(dftrainStat, dftrainStat, 1, 6, colLst, '3')
dftrainStat = genStat(dftrainStat, dftrainStat, 1, 12, colLst, '3')
dftrainStat = genStat(dftrainStat, dftrainStat, 1, 18, colLst, '3')

colLst = ['bedroomcnt','regionidcity']

dftrainStat = genStat(dftrainStat, dftrainStat, 1, 3, colLst, '4')
dftrainStat = genStat(dftrainStat, dftrainStat, 1, 6, colLst, '4')
dftrainStat = genStat(dftrainStat, dftrainStat, 1, 12, colLst, '4')
dftrainStat = genStat(dftrainStat, dftrainStat, 1, 18, colLst, '4')

y = dftrainStat.logerror.values
joblib.dump(y, "../input/y.pkl")
trainparcel = dftrainStat.parcelid.values
joblib.dump(trainparcel, "../input/trainparcel.pkl")
dftrainStat.drop(['logerror', 'parcelid'], axis = 1, inplace = True)

print(dftrainStat.shape)
print(dftrainStat.head())
print(dftrainStat.loc[0])

joblib.dump(dftrainStat,"../input/trainstat.pkl")

colLst = ['bedroomcnt','censustractandblock']

dftrainStat = genStat(train, test, 1, 3, colLst)
dftrainStat = genStat(train, dftrainStat, 1, 6, colLst)
dftrainStat = genStat(train, dftrainStat, 1, 12, colLst)
dftrainStat = genStat(train, dftrainStat, 1, 18, colLst)

colLst = ['bedroomcnt','regionidneighborhood']

dftrainStat = genStat(train, dftrainStat, 1, 3, colLst, '2')
dftrainStat = genStat(train, dftrainStat, 1, 6, colLst, '2')
dftrainStat = genStat(train, dftrainStat, 1, 12, colLst, '2')
dftrainStat = genStat(train, dftrainStat, 1, 18, colLst, '2')

colLst = ['bedroomcnt','regionidzip']
dftrainStat = genStat(train, dftrainStat, 1, 3, colLst, '3')
dftrainStat = genStat(train, dftrainStat, 1, 6, colLst, '3')
dftrainStat = genStat(train, dftrainStat, 1, 12, colLst, '3')
dftrainStat = genStat(train, dftrainStat, 1, 18, colLst, '3')

colLst = ['bedroomcnt','regionidcity']

dftrainStat = genStat(train, dftrainStat, 1, 3, colLst, '4')
dftrainStat = genStat(train, dftrainStat, 1, 6, colLst, '4')
dftrainStat = genStat(train, dftrainStat, 1, 12, colLst, '4')
dftrainStat = genStat(train, dftrainStat, 1, 18, colLst, '4')


testparcelid = dftrainStat.parcelid
joblib.dump(testparcelid, '../input/testparcel.pkl')
dftrainStat.drop(['parcelid'], axis = 1, inplace = True)

print(dftrainStat.shape)
print(dftrainStat.head())
print(dftrainStat.loc[0])

joblib.dump(dftrainStat,"../input/teststat.pkl")


