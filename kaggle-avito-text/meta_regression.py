import os; os.environ['OMP_NUM_THREADS'] = '1'
import nltk
nltk.data.path.append("/media/sayantan/Personal/nltk_data")
from nltk.stem.snowball import RussianStemmer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer as Tfidf
from sklearn.metrics import mean_squared_log_error
from sklearn.model_selection import KFold
from sklearn.externals import joblib
from scipy.sparse import hstack, csr_matrix
import lightgbm as lgb
from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np
import tensorflow as tf
import gc
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder, MinMaxScaler
from sklearn import model_selection
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import Ridge, Lasso, HuberRegressor, ElasticNet, BayesianRidge, LinearRegression
region = joblib.load("../region_onehot.pkl")
city = joblib.load("../city_onehot.pkl")
parent_category_name = joblib.load("../parent_category_name_onehot.pkl")
category_name = joblib.load("../category_name_onehot.pkl")
user_type = joblib.load("../user_type_onehot.pkl")

train_df = pd.read_feather('../train_basic_features_woCats.pkl')
test_df = pd.read_feather('../test__basic_features_woCats.pkl')
y = train_df.deal_probability.values
test_id = test_df.item_id.values
train_df.drop(['deal_probability'], axis = 'columns', inplace = True)
test_df.drop(['deal_probability'], axis = 'columns', inplace = True)
item_id = test_df.item_id.values

cols_to_drop =["item_id", "title", "description", "activation_date", "image", "params",
               "title_clean", "desc_clean", "params_clean",
               "get_nouns_title","get_nouns_desc",
               "get_adj_title","get_adj_desc",
               "get_verb_title","get_verb_desc",
               "monthday", "price"
               ]

train_df.drop(cols_to_drop, axis = 'columns', inplace = True)
test_df.drop(cols_to_drop, axis = 'columns', inplace = True)
gc.collect()
#train_df.fillna(-1, inplace = True)
#test_df.fillna(-1, inplace = True)

for col in train_df.columns:
    lbl = MinMaxScaler()
    X = np.hstack((train_df[col].fillna(-1).values, test_df[col].fillna(-1).values)).reshape(-1,1)
    lbl.fit(X)
    train_df[col] = lbl.transform(train_df[col].fillna(-1).values.reshape(-1,1))
    test_df[col] = lbl.transform(test_df[col].fillna(-1).values.reshape(-1,1))

week = joblib.load("../activation_weekday_onehot.pkl")
train_df.drop(['activation_weekday'], axis = 'columns', inplace = True)
test_df.drop(['activation_weekday'], axis = 'columns', inplace = True)

param_train_tfidf, param_test_tfidf = joblib.load("../params_tfidf2.pkl")
title_train_tfidf, title_test_tfidf = joblib.load("../title_tfidf2.pkl")
desc_train_tfidf, desc_test_tfidf = joblib.load("../desc_tfidf.pkl")

#nouns_title_train_tfidf, nouns_title_test_tfidf = joblib.load("../nouns_title_tfidf.pkl")
#nouns_desc_train_tfidf, nouns_desc_test_tfidf = joblib.load("../nouns_desc_tfidf.pkl")
#adj_title_train_tfidf, adj_title_test_tfidf = joblib.load("../adj_title_tfidf.pkl")
#adj_desc_train_tfidf, adj_desc_test_tfidf = joblib.load("../adj_desc_tfidf.pkl")
#verb_title_train_tfidf, verb_title_test_tfidf = joblib.load("../verb_title_tfidf.pkl")
#verb_desc_train_tfidf, verb_desc_test_tfidf = joblib.load("../verb_desc_tfidf.pkl")

region_train,region_test = region[:train_df.shape[0]],region[train_df.shape[0]:]
pcn_train, pcn_test = parent_category_name[:train_df.shape[0]], parent_category_name[train_df.shape[0]:] 
cn_train, cn_test = category_name[:train_df.shape[0]],category_name[train_df.shape[0]:],  
ut_train, ut_test = user_type[:train_df.shape[0]], user_type[train_df.shape[0]:]
city_train, city_test = city[:train_df.shape[0]], city[train_df.shape[0]:]
week_train, week_test = week[:train_df.shape[0]],week[train_df.shape[0]:]


param_train_tfidf, param_test_tfidf = param_train_tfidf.tocsr(), param_test_tfidf.tocsr()
title_train_tfidf, title_test_tfidf = title_train_tfidf.tocsr(), title_test_tfidf.tocsr()
desc_train_tfidf, desc_test_tfidf = desc_train_tfidf.tocsr(), desc_test_tfidf.tocsr()

tit_train_char, tit_test_char = joblib.load('../title_chargram_tfidf.pkl')
par_train_char, par_test_char = joblib.load('../param_chargram_tfidf.pkl')
cat_train_char, cat_test_char = joblib.load('../cat_chargram_tfidf.pkl')

train_2=pd.read_feather('../train_cat_targetenc.pkl')
test_2=pd.read_feather('../test_cat_targetenc.pkl')
catCols = ['user_id', 'region', 'city', 'parent_category_name',
       'category_name', 'user_type']
train_2.drop(catCols, axis = 'columns', inplace=True)
test_2.drop(catCols, axis = 'columns', inplace=True)

#train_2.fillna(-1, inplace = True)
#test_2.fillna(-1, inplace = True)

for col in train_2.columns:
    lbl = MinMaxScaler()
    X = np.hstack((train_2[col].fillna(-1).values, test_2[col].fillna(-1).values)).reshape(-1,1)
    lbl.fit(X)
    train_2[col] = lbl.transform(train_2[col].fillna(-1).values.reshape(-1,1))
    test_2[col] = lbl.transform(test_2[col].fillna(-1).values.reshape(-1,1))

train_3=pd.read_feather('../train_kag_agg_ftr.ftr')
test_3=pd.read_feather('../test_kag_agg_ftr.ftr')

for col in train_3.columns:
    lbl = MinMaxScaler()
    X = np.hstack((train_3[col].fillna(-1).values, test_3[col].fillna(-1).values)).reshape(-1,1)
    lbl.fit(X)
    train_3[col] = lbl.transform(train_3[col].fillna(-1).values.reshape(-1,1))
    test_3[col] = lbl.transform(test_3[col].fillna(-1).values.reshape(-1,1))


i = 0
nbag = 1
nfold = 7
train_df = hstack([train_df.values,param_train_tfidf, title_train_tfidf, desc_train_tfidf,
                   region_train, pcn_train, cn_train, ut_train, city_train, week_train, train_2, 
                   tit_train_char, par_train_char,
                   cat_train_char, train_3
                   ]) # Sparse Matrix
train_df = train_df.tocsr()

del(param_train_tfidf, title_train_tfidf, desc_train_tfidf,
                   region_train, pcn_train, cn_train, ut_train, city_train, week_train, train_2, 
                   tit_train_char, par_train_char,
                   cat_train_char, train_3
                   ); gc.collect()


test_df = hstack([test_df.values,param_test_tfidf, title_test_tfidf, desc_test_tfidf,
                   region_test, pcn_test, cn_test, ut_test, city_test, week_test, test_2, 
                   tit_test_char, par_test_char,
                   cat_test_char, test_3
                   ]) # Sparse Matrix
test_df =test_df.tocsr()


del(param_test_tfidf, title_test_tfidf, desc_test_tfidf,
                   region_test, pcn_test, cn_test, ut_test, city_test, week_test, test_2, 
                   tit_test_char, par_test_char,
                   cat_test_char, test_3
                   )                    
gc.collect()

oobtest = np.zeros((test_df.shape[0],4))
oobval = np.zeros((train_df.shape[0],4))
valerr = []
cnt = 0
cv_r2 = []
nfold = 7
nbag =1
np.random.seed(2018)
for seed in [2018]:
    kf = model_selection.KFold(n_splits=nfold, shuffle=False, random_state=seed)
    for dev_index, val_index in kf.split(y): 
        dev_X, val_X = train_df[dev_index,:], train_df[val_index,:]
        dev_y, val_y = y[dev_index], y[val_index]
        print(dev_X.shape)  

        ir = HuberRegressor()
        ir.fit(dev_X, dev_y)
        preds = ir.predict(val_X)
        oobval[val_index,0] += preds
        cv_r2.append(mean_squared_error(val_y, preds) ** 0.5)    
        print(cv_r2, np.mean(cv_r2),"---", np.std(cv_r2))
        predtst = ir.predict(test_df)
        oobtest[:,0] += predtst

        ir = Ridge(alpha = 0.01)
        ir.fit(dev_X, dev_y)
        preds = ir.predict(val_X)
        oobval[val_index,1] += preds
        cv_r2.append(mean_squared_error(val_y, preds) ** 0.5)    
        print(cv_r2, np.mean(cv_r2),"---", np.std(cv_r2))
        predtst = ir.predict(test_df)
        oobtest[:,1] += predtst

        ir = Ridge(alpha = 1.0)
        ir.fit(dev_X, dev_y)
        preds = ir.predict(val_X)
        oobval[val_index,2] += preds
        cv_r2.append(mean_squared_error(val_y, preds) ** 0.5)    
        print(cv_r2, np.mean(cv_r2),"---", np.std(cv_r2))
        predtst = ir.predict(test_df)
        oobtest[:,2] += predtst

        ir = Ridge(alpha = 10.0)
        ir.fit(dev_X, dev_y)
        preds = ir.predict(val_X)
        oobval[val_index,3] += preds
        cv_r2.append(mean_squared_error(val_y, preds) ** 0.5)    
        print(cv_r2, np.mean(cv_r2),"---", np.std(cv_r2))
        predtst = ir.predict(test_df)
        oobtest[:,3] += predtst

        del(dev_X, val_X); gc.collect()
tstpred = oobtest / (nfold * nbag)
oobpred = oobval / nbag
oobpred[oobpred>1] = 1
oobpred[oobpred<0] = 0

tstpred[tstpred>1] = 1
tstpred[tstpred<0] = 0

joblib.dump(oobpred,'../l1-train_weakregr.pkl')
joblib.dump(tstpred,'../l1-test_weakregr.pkl')
           
####################################

