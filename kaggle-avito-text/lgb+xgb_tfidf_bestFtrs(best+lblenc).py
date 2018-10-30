import os; os.environ['OMP_NUM_THREADS'] = '3'
from sklearn.ensemble import ExtraTreesRegressor
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
import xgboost as xgb
from sklearn.ensemble import ExtraTreesRegressor

train_x1 = pd.read_feather('../train_imagetop_targetenc.pkl')
train_x2 = pd.read_feather('../train_itemseq_targetenc.pkl')

test_x1 = pd.read_feather('../test_imagetop_targetenc.pkl')
test_x2 = pd.read_feather('../test_itemseq_targetenc.pkl')

#region = joblib.load("../region_onehot.pkl")
#city = joblib.load("../city_onehot.pkl")
#parent_category_name = joblib.load("../parent_category_name_onehot.pkl")
#category_name = joblib.load("../category_name_onehot.pkl")
#user_type = joblib.load("../user_type_onehot.pkl")

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

#for col in train_df.columns:
#    lbl = MinMaxScaler()
#    X = np.hstack((train_df[col].fillna(-1).values, test_df[col].fillna(-1).values)).reshape(-1,1)
#    lbl.fit(X)
#    train_df[col] = lbl.transform(train_df[col].fillna(-1).values.reshape(-1,1))
#    test_df[col] = lbl.transform(test_df[col].fillna(-1).values.reshape(-1,1))

week = joblib.load("../activation_weekday_onehot.pkl")
train_df.drop(['activation_weekday'], axis = 'columns', inplace = True)
test_df.drop(['activation_weekday'], axis = 'columns', inplace = True)

param_train_tfidf, param_test_tfidf = joblib.load("../params_tfidf.pkl")
title_train_tfidf, title_test_tfidf = joblib.load("../title_tfidf.pkl")
desc_train_tfidf, desc_test_tfidf = joblib.load("../desc_tfidf.pkl")

#nouns_title_train_tfidf, nouns_title_test_tfidf = joblib.load("../nouns_title_tfidf.pkl")
#nouns_desc_train_tfidf, nouns_desc_test_tfidf = joblib.load("../nouns_desc_tfidf.pkl")
#adj_title_train_tfidf, adj_title_test_tfidf = joblib.load("../adj_title_tfidf.pkl")
#adj_desc_train_tfidf, adj_desc_test_tfidf = joblib.load("../adj_desc_tfidf.pkl")
#verb_title_train_tfidf, verb_title_test_tfidf = joblib.load("../verb_title_tfidf.pkl")
#verb_desc_train_tfidf, verb_desc_test_tfidf = joblib.load("../verb_desc_tfidf.pkl")

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

train_df2=pd.read_feather('../train_basic_features_lblencCats.pkl')
test_df2=pd.read_feather('../test__basic_features_lblencCats.pkl')
catCols = ['user_id', 'region', 'city', 'parent_category_name',
       'category_name', 'user_type', 'param_1', 'param_2', 'param_3']
train_df2 = train_df2[catCols]
test_df2 = test_df2[catCols]
region_train,region_test = train_df2.region.values.reshape(-1,1),test_df2.region.values.reshape(-1,1)
pcn_train, pcn_test = train_df2.parent_category_name.values.reshape(-1,1), test_df2.parent_category_name.values.reshape(-1,1)
cn_train, cn_test = train_df2.category_name.values.reshape(-1,1), test_df2.category_name.values .reshape(-1,1)
ut_train, ut_test = train_df2.user_type.values.reshape(-1,1), test_df2.user_type.values .reshape(-1,1)
city_train, city_test = train_df2.city.values.reshape(-1,1), test_df2.city.values .reshape(-1,1)

del(train_df2, test_df2);gc.collect()

train_3=pd.read_feather('../train_kag_agg_ftr.ftr')
test_3=pd.read_feather('../test_kag_agg_ftr.ftr')

#for col in train_3.columns:
#    lbl = MinMaxScaler()
#    X = np.hstack((train_3[col].fillna(-1).values, test_3[col].fillna(-1).values)).reshape(-1,1)
#    lbl.fit(X)
#    train_3[col] = lbl.transform(train_3[col].fillna(-1).values.reshape(-1,1))
#    test_3[col] = lbl.transform(test_3[col].fillna(-1).values.reshape(-1,1))


train_4 = joblib.load('../l1-train_weakregr.pkl')
test_4 = joblib.load('../l1-test_weakregr.pkl')


train_df = hstack([train_df.values,param_train_tfidf, title_train_tfidf, desc_train_tfidf,
                   region_train, pcn_train, cn_train, ut_train, city_train, week_train, train_2, 
                   tit_train_char, par_train_char,
                   cat_train_char, train_3, train_4,
                   train_x1.values, train_x2.values
                   ]) # Sparse Matrix
train_df = train_df.tocsr()

del(param_train_tfidf, title_train_tfidf, desc_train_tfidf,
                   region_train, pcn_train, cn_train, ut_train, city_train, week_train, train_2, 
                   tit_train_char, par_train_char,
                   cat_train_char, train_3, train_4,
                   train_x1, train_x2                   
                   ); gc.collect()


test_df = hstack([test_df.values,param_test_tfidf, title_test_tfidf, desc_test_tfidf,
                   region_test, pcn_test, cn_test, ut_test, city_test, week_test, test_2, 
                   tit_test_char, par_test_char,
                   cat_test_char, test_3, test_4,
                   test_x1.values, test_x2.values                   
                   ]) # Sparse Matrix
test_df =test_df.tocsr()


del(param_test_tfidf, title_test_tfidf, desc_test_tfidf,
                   region_test, pcn_test, cn_test, ut_test, city_test, week_test, test_2, 
                   tit_test_char, par_test_char,
                   cat_test_char, test_3, test_4,
                   test_x1, test_x2
                   ) 
                   
gc.collect()

oobtest = np.zeros((test_df.shape[0],2))
oobval = np.zeros((train_df.shape[0],2))
valerr = []
cnt = 0
cv_r2 = []
nfold = 7
nbag =1
np.random.seed(2018)
for i, n  in [[15,4500]]:#, [14, 4800], [16,4300]]: 
    for seed in [2018]:
        kf = model_selection.KFold(n_splits=nfold, shuffle=False, random_state=seed)
        for dev_index, val_index in kf.split(y): 
            dev_X, val_X = train_df[dev_index,:], train_df[val_index,:]
            dev_y, val_y = y[dev_index], y[val_index]
            print(dev_X.shape)  
            #break
            print("Light Gradient Boosting Regressor")

            # LGBM Dataset Formatting 
            lgtrain = lgb.Dataset(dev_X, dev_y)
            lgvalid = lgb.Dataset(val_X, val_y)

            # Train
            lgbm_params =  {
                'task': 'train',
                'boosting_type': 'gbdt',
                'objective': 'regression',
                'metric': 'rmse',
                'max_depth': i,
                'num_leaves': 150,#31
                'feature_fraction': 0.65,
                'bagging_fraction': 0.9,
                'bagging_freq': 5,
                'learning_rate': 0.05,
                'verbose': 0,
                'feature_fraction_seed': seed,
                'bagging_seed': seed
            }  
            model = lgb.train(
                lgbm_params,
                lgtrain,
                num_boost_round=n,
                valid_sets=[lgtrain, lgvalid],
                valid_names=['train','valid'],
                early_stopping_rounds=200,
                verbose_eval=200
            )
#[1731]	train's rmse: 0.196296	valid's rmse: 0.214809

            preds = model.predict(val_X, num_iteration=model.best_iteration).reshape(-1,1)
            oobval[val_index,0:1] += preds
            cv_r2.append(mean_squared_error(val_y, preds) ** 0.5)    
            print(cv_r2, np.mean(cv_r2),"---", np.std(cv_r2))
            predtst = model.predict(test_df, num_iteration=model.best_iteration).reshape(-1,1)
            oobtest[:,0:1] += predtst
            del(lgtrain, lgvalid);gc.collect()            
            print("XGB Regressor")
            param = {
                'eta': 0.05,
                #'tree_method':'hist',
                'objective': 'reg:linear',
                'eval_metric': 'rmse',
                'base_score': dev_y.mean(),
                'silent': 1,
                'seed' : seed,    
                'subsample': .9,
                'min_child_weight' : .5,
                'colsample_bytree' :0.75,
                'alpha' : 2.0,
                'lambda' : 3.0,
                'gamma' : 0.01,
                'colsample_bylevel' : 0.95,
                'max_depth': i
            }

            # Dataset Formatting 
            xgtrain = xgb.DMatrix(dev_X, label=dev_y, missing = -999)
            xgval = xgb.DMatrix(val_X, label=val_y, missing = -999)

            # Train
            watchlist = [ (xgtrain,'train'), (xgval, 'val') ] 
            model = xgb.train(param, xgtrain, 700, watchlist, verbose_eval=100, early_stopping_rounds=100) #n
            xgval = xgb.DMatrix(val_X)

            preds = model.predict(xgval, ntree_limit=model.best_ntree_limit).reshape(-1,1)
            oobval[val_index,1:2] += preds
            cv_r2.append(mean_squared_error(val_y, preds) ** 0.5)    
            print(cv_r2, np.mean(cv_r2),"---", np.std(cv_r2))
            predtst = model.predict(xgb.DMatrix(test_df, missing = -999), ntree_limit=model.best_ntree_limit).reshape(-1,1)
            oobtest[:,1:2] += predtst
            del(xgval, xgtrain); gc.collect()           
            del(dev_X, val_X); gc.collect()
            #break
tstpred = oobtest / (nfold * nbag)#*3
oobpred = oobval / (nbag)    #*3)
oobpred[oobpred>1] = 1
oobpred[oobpred<0] = 0

tstpred[tstpred>1] = 1
tstpred[tstpred<0] = 0

joblib.dump(oobpred,'../l1-train_lgb_bstFtr(tfidf2+lblenc)_7f.pkl')
joblib.dump(tstpred,'../l1-test_lgb_bstFtr(tfidf2+lblenc)_7f.pkl')

# Making a submission file #
tstpred = (tstpred[:,0]+tstpred[:,1])/2
sub_df = pd.DataFrame({"item_id":test_id})
sub_df["deal_probability"] = tstpred
sub_df.to_csv("../output/l1-train_xgb+lgb_bstFtr(tfidf2+lblenc)_7f.csv", index=False)    
#PLB -2220
# Making a submission file #
tstpred1 = (tstpred[:,0])/1
sub_df = pd.DataFrame({"item_id":test_id})
sub_df["deal_probability"] = tstpred1
sub_df.to_csv("../output/l1-train_xgb+lgb_bstFtr(tfidf2+lblenc)_7f_1.csv", index=False)    
#plb 2222
# Making a submission file #
tstpred = (tstpred[:,1])/1
sub_df = pd.DataFrame({"item_id":test_id})
sub_df["deal_probability"] = tstpred
sub_df.to_csv("../output/l1-train_xgb+lgb_bstFtr(tfidf2+lblenc)_7f_2.csv", index=False)    
# plb 2223


