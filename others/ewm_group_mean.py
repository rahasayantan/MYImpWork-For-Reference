import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn import preprocessing, model_selection, metrics
import lightgbm as lgb
import gc

train_df = pd.read_csv('../input/train.csv',  parse_dates=["activation_date"])
test_df = pd.read_csv('../input/test.csv',  parse_dates=["activation_date"])

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np 
import seaborn as sns
import random 

#x = train_df.copy()#[['price', 'deal_probability', 'image_top_1']]

def genFeatures(x):
    x["activation_weekday"] = x["activation_date"].dt.weekday
    x["monthday"] = x["activation_date"].dt.day
##################Added in set 1 - 0.01 Improvement
    x['price_new'] = np.log1p(x.price) # log transform improves co-relation with deal_price
    x['count_null_in_row'] = x.isnull().sum(axis=1)# works
    x['has_description'] = x.description.isnull().astype(int) 
    x['has_image'] = x.image.isnull().astype(int) 
    x['has_image_top'] = x.image_top_1.isnull().astype(int) 
    x['has_param1'] = x.param_1.isnull().astype(int) 
    x['has_param2'] = x.param_2.isnull().astype(int) 
    x['has_param3'] = x.param_3.isnull().astype(int) 
    x['has_price'] = x.price.isnull().astype(int) 
#################Added in set 2 - 0.00x Improvement
    x["description"].fillna("NA", inplace=True)
    x["desc_nwords"] = x["description"].apply(lambda x: len(x.split()))
    x['len_description'] = x['description'].apply(lambda x: len(x))
    x["title_nwords"] = x["title"].apply(lambda x: len(x.split()))   
    x['len_title'] = x['title'].apply(lambda x: len(x))
    x['params'] = x['param_1'].fillna('') + ' ' + x['param_2'].fillna('') + ' ' + x['param_3'].fillna('')
    x['params'] = x['params'].str.strip()
    x['len_params'] = x['params'].apply(lambda x: len(x))
    x['words_params'] = x['params'].apply(lambda x: len(x.split()))
    x['symbol1_count'] = x['description'].str.count('↓')
    x['symbol2_count'] = x['description'].str.count('\*')
    x['symbol3_count'] = x['description'].str.count('✔')
    x['symbol4_count'] = x['description'].str.count('❀')
    x['symbol5_count'] = x['description'].str.count('➚')
    x['symbol6_count'] = x['description'].str.count('ஜ')
    x['symbol7_count'] = x['description'].str.count('.')
    x['symbol8_count'] = x['description'].str.count('!')
    x['symbol9_count'] = x['description'].str.count('\?')
    x['symbol10_count'] = x['description'].str.count('  ')
    x['symbol11_count'] = x['description'].str.count('-')
    x['symbol12_count'] = x['description'].str.count(',') 
####################    
    return x
    
train_df = genFeatures(train_df)
test_df = genFeatures(test_df)

groupCols = ['region', 'city', 'parent_category_name',
       'category_name', 'user_type']
X = train_df[groupCols + ['deal_probability']].groupby(groupCols, as_index=False).agg([len,np.mean])
X.columns = ['_'.join(col).strip() for col in X.columns.values]
X['Group_weight1'] = (X.deal_probability_mean + 1e-6) * np.log1p(X.deal_probability_len) 
X.drop(['deal_probability_mean', 'deal_probability_len'], axis = 1, inplace = True)
X.reset_index(inplace = True)
train_df = train_df.merge(X, on = groupCols, how = 'left')
test_df = test_df.merge(X, on = groupCols, how = 'left')

def catEncode(train_char, test_char, y, colLst = [], nbag = 10, nfold = 20, minCount = 3, val = False, val_char = None):
    train_df = train_char.copy()
    test_df = test_char.copy()
    if val == True:
        val_df = val_char.copy()
    if not colLst:
        print("Empty ColLst")
        for c in train_char.columns:
            data = train_char[[c]].copy()
            data['y'] = y
            enc_mat = np.zeros((y.shape[0],3))
            enc_mat_test = np.zeros((test_char.shape[0],3))
            if val ==True:
                enc_mat_val = np.zeros((val_char.shape[0],3))            
            for bag in np.arange(nbag):
                kf = model_selection.KFold(n_splits= nfold, shuffle=True, random_state=2017*bag)
                for dev_index, val_index in kf.split(range(data['y'].shape[0])):
                    dev_X, val_X = data.iloc[dev_index,:], data.iloc[val_index,:]
                    datax = dev_X.groupby([c]).agg([len,np.mean,np.std])
                    datax.columns = ['_'.join(col).strip() for col in datax.columns.values]
                    datax = datax.loc[datax.y_len > minCount]
                    ind = c 
                    datax.rename(columns = {'y_mean': ('y_mean_' + ind), 'y_std': ('y_std_' + ind),
                    'y_len': ('y_len_' + ind)}, inplace = True)
#                    datax[c+'_medshftenc'] =  datax['y_median']-med_y
#                    datax.drop(['y_len','y_mean','y_std','y_median'],axis=1,inplace=True)
                    datatst = test_char[[c]].copy()
                    val_X = val_X.join(datax,on=[c], how='left').fillna(-99)
                    datatst = datatst.join(datax,on=[c], how='left').fillna(-99)
                    enc_mat[val_index,...] += val_X[list(set(datax.columns)-set([c]))]
                    enc_mat_test += datatst[list(set(datax.columns)-set([c]))]
                    if val ==True:
                        valTst = val_char[[c]].copy()
                        valTst = valTst.join(datax,on=[c], how='left').fillna(-99)                        
                        enc_mat_val += valTst[list(set(datax.columns)-set([c]))]          
            enc_mat_test /= (nfold * nbag)
            enc_mat /= (nbag)        
            enc_mat = pd.DataFrame(enc_mat)  
            enc_mat.columns = list(set(datax.columns)-set([c]))
            enc_mat_test = pd.DataFrame(enc_mat_test)  
            enc_mat_test.columns=enc_mat.columns
            if val == True:
                enc_mat_val /= (nfold * nbag)
                enc_mat_val = pd.DataFrame(enc_mat_val)  
                enc_mat_val.columns=enc_mat.columns
                val_df = pd.concat([enc_mat_val.reset_index(drop = True),val_df.reset_index(drop = True)],axis=1)
            train_df = pd.concat((enc_mat.reset_index(drop = True),train_df.reset_index(drop = True)), axis=1)
            test_df = pd.concat([enc_mat_test.reset_index(drop = True),test_df.reset_index(drop = True)],axis=1)
        
    else:
        print("Not Empty ColLst")
        data = train_char[colLst].copy()
        data['y'] = y
        enc_mat = np.zeros((y.shape[0],3))
        enc_mat_test = np.zeros((test_char.shape[0],3))
        if val ==True:
            enc_mat_val = np.zeros((val_char.shape[0],3))            
        for bag in np.arange(nbag):     
            kf = model_selection.KFold(n_splits= nfold, shuffle=True, random_state=2017*bag)
            for dev_index, val_index in kf.split(range(data['y'].shape[0])):
                dev_X, val_X = data.iloc[dev_index,:], data.iloc[val_index,:]
                datax = dev_X.groupby(colLst).agg([len,np.mean,np.std])
                datax.columns = ['_'.join(col).strip() for col in datax.columns.values]
                datax = datax.loc[datax.y_len > minCount]
                ind = '_'.join(colLst)
                datax.rename(columns = {'y_mean': ('y_mean_' + ind), 'y_std': ('y_std_' + ind),
                'y_len': ('y_len_' + ind)}, inplace = True)
                datatst = test_char[colLst].copy()
                val_X = val_X.join(datax,on=colLst, how='left').fillna(-99)
                datatst = datatst.join(datax,on=colLst, how='left').fillna(-99)
                print(val_X[list(set(datax.columns)-set(colLst))].columns)
                enc_mat[val_index,...] += val_X[list(set(datax.columns)-set(colLst))]
                enc_mat_test += datatst[list(set(datax.columns)-set(colLst))]
                if val ==True:
                    valTst = val_char[[c]].copy()
                    valTst = valTst.join(datax,on=[c], how='left').fillna(-99)                        
                    enc_mat_val += valTst[list(set(datax.columns)-set([c]))]          
                
        enc_mat_test /= (nfold * nbag)
        enc_mat /= (nbag)        
        enc_mat = pd.DataFrame(enc_mat)  
        enc_mat.columns=list(set(datax.columns)-set([c]))
        enc_mat_test = pd.DataFrame(enc_mat_test)  
        enc_mat_test.columns=enc_mat.columns 
        train_df = pd.concat((enc_mat.reset_index(drop = True),train_df.reset_index(drop = True)), axis=1)
        test_df = pd.concat([enc_mat_test.reset_index(drop = True),test_df.reset_index(drop = True)],axis=1)
        if val == True:
            enc_mat_val /= (nfold * nbag)
            enc_mat_val = pd.DataFrame(enc_mat_val)  
            enc_mat_val.columns=enc_mat.columns
            val_df = pd.concat([enc_mat_val.reset_index(drop = True),val_df.reset_index(drop = True)],axis=1)
    print(train_df.columns)
    print(test_df.columns) 
    if val == True:
        print(val_df.columns)                           
    for c in train_df.columns:
        if train_df[c].dtype == 'float64':
            train_df[c] = train_df[c].astype('float32')
            test_df[c] = test_df[c].astype('float32')
    if val == True:
        for c in train_df.columns:
            if train_df[c].dtype == 'float64':
                train_df[c] = train_df[c].astype('float32')
                test_df[c] = test_df[c].astype('float32')
                val_df[c] = val_df[c].astype('float32')
        return train_df, test_df, val_df
    else:
        return train_df, test_df


catCols = ['region', 'city', 'parent_category_name',
       'category_name', 'param_1', 'param_2', 'param_3', 'user_type']

dftrainnum = train_df[list(set(train_df.columns)-set(catCols+['user_id']))]
dftestnum = test_df[list(set(test_df.columns)-set(catCols+['user_id']))]

train, test,= = catEncode(train_df[catCols].copy(), test_df[catCols].copy(), train_df.deal_probability.values, nbag = 10, nfold = 20, minCount = 1)

train_df = pd.concat((dftrainnum, train), axis =1)
test_df = pd.concat((dftestnum, test), axis =1)

del(dftrainnum, train); gc.collect()
del(dftestnum, test); gc.collect()




