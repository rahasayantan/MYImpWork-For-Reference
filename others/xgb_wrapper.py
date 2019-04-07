from sklearn.externals import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn import model_selection, preprocessing, ensemble
from sklearn.metrics import mean_absolute_error
import gc
import math
from collections import Counter
from bayes_opt import BayesianOptimization
from matplotlib import pylab as plt
import operator


class xgboostModel(object):
    def __init__(self, params):
        self.params = params
        self.model = None
        self.pred = None
        self.dev_X = None
        self.val_X = None
        self.val_y = None
        self.dev_y = None
        self.y = None
    def train(self, train, y, val = None, valy = None, validate = True, num_rounds = 100000, custEval = False, early_stopping_rounds=200, verbose_eval=100):
        plst = list(self.params.items())
        def xgb_mean_absolute_error( preds, dtrain):
            labels = dtrain.get_label()
            return 'r2', mean_absolute_error(labels, preds)
        xgtrain = xgb.DMatrix(train, label=y)
        if validate == True:
            num_rounds = num_rounds
            xgval = xgb.DMatrix(val, label=valy)
            watchlist = [ (xgtrain,'train'), (xgval, 'val') ]
            if custEval == False:
                self.model = xgb.train(plst, xgtrain, num_rounds, watchlist, early_stopping_rounds=early_stopping_rounds, verbose_eval=100)
#                best_iteration = self.model.best_iteration+1
#                self.model = xgb.train(plst, xgtrain, best_iteration, watchlist, early_stopping_rounds=early_stopping_rounds, verbose_eval=100)
            else:
                self.model = xgb.train(plst, xgtrain, num_rounds, watchlist, feval=xgb_mean_absolute_error, maximize=True,early_stopping_rounds=early_stopping_rounds, verbose_eval=100)
#                best_iteration =self. model.best_iteration+1
#                self.model = xgb.train(plst, xgtrain, best_iteration, watchlist, feval=xgb_mean_absolute_error, maximize=True,early_stopping_rounds=early_stopping_rounds, verbose_eval=100) 
        else:
            watchlist = [ (xgtrain,'train'), (xgtrain,'val')]
            if custEval == False:        
                self.model = xgb.train(plst, xgtrain, num_rounds, watchlist, verbose_eval=verbose_eval)
            else:
                self.model = xgb.train(plst, xgtrain, num_rounds, watchlist, feval=xgb_mean_absolute_error, maximize=True, verbose_eval=verbose_eval)                

    def predict(self, test, predCol = 1):
        xgtest = xgb.DMatrix(test)   
        self.pred = self.model.predict(xgtest, ntree_limit=self.model.best_ntree_limit)
        return self.pred.reshape(-1,predCol)

    def custTrainSplitPrediction(self, trainx, valx, trainy, valy, test_df, y, nbag = 5, seed = 201706, predCol = 1):
        cv_scores = []
        cv_r2 = []
        dev_X, val_X, dev_y, val_y = trainx, valx, trainy, valy
        #print(dev_y)
        oob_valpred = np.zeros((val_X.shape[0],predCol))
        oob_tstpred = np.zeros((test_df.shape[0],predCol))
        
        for nbagi in np.arange(nbag):
            self.params['seed'] = nbagi*seed+231
            self.train(dev_X, dev_y, val_X, val_y)
            preds = self.predict(val_X,predCol)
            
            oob_valpred += preds
            cv_scores.append(mean_absolute_error(val_y, preds))
            if predCol == 1:
                cv_r2.append(mean_absolute_error(val_y, preds))    
                print(mean_absolute_error(val_y, preds))
                print(cv_r2)
                print(np.mean(cv_r2))
            predtst = self.predict(test_df,predCol)
            oob_tstpred += predtst
        oob_tstpred /= (nbag)
        oob_valpred /= (nbag) 
        return oob_tstpred, oob_valpred, cv_scores, cv_r2

    
    def trainSplitPrediction(self, train_df, test_df, y, nbag = 5, seed = 201706, test_size = 0.2, stratify = None, predCol = 1):
        cv_scores = []
        cv_r2 = []
        dev_X, val_X, dev_y, val_y = model_selection.train_test_split(train_df, y, test_size=test_size, random_state=seed, stratify = stratify)    
        #print(dev_y)
        oob_valpred = np.zeros((val_X.shape[0],predCol))
        oob_tstpred = np.zeros((test_df.shape[0],predCol))
        
        for nbagi in np.arange(nbag):
            dev_X, val_X, dev_y, val_y = model_selection.train_test_split(train_df, y, test_size=test_size, random_state=seed, stratify = stratify)    
            self.params['seed'] = nbagi*201706+231
            self.train(dev_X, dev_y, val_X, val_y)
            preds = self.predict(val_X,predCol)
            
            oob_valpred += preds
            cv_scores.append(mean_absolute_error(val_y, preds))
            if predCol == 1:
                cv_r2.append(mean_absolute_error(val_y, preds))    
                print(mean_absolute_error(val_y, preds))
                print(cv_r2)
                print(np.mean(cv_r2))
            predtst = self.predict(test_df,predCol)
            oob_tstpred += predtst
        oob_tstpred /= (nbag)
        oob_valpred /= (nbag) 
        return oob_tstpred, oob_valpred, cv_scores, cv_r2
    
    def oofPrediction(self, train_df, test_df, y, y_logit = None, nfold = 5, nbag = 1, seed = 201706, custEval = False, predCol = 1):
        cv_scores = []
        cv_r2 = []
        oob_valpred = np.zeros((train_df.shape[0],predCol))
        oob_tstpred = np.zeros((test_df.shape[0],predCol))
        
        for nbagi in np.arange(nbag):
            if list(y_logit) == None:
                y_logit = y
                kf = model_selection.KFold(n_splits=nfold, shuffle=False, random_state=nbagi*seed+231)
            else:
                kf = model_selection.StratifiedKFold(n_splits=nfold, shuffle=False, random_state=nbagi*seed+231)                
            for dev_index, val_index in kf.split(train_df,y_logit): # explain for regression convert y to bins and use that for split
                if isinstance(train_df, np.ndarray):
                    dev_X, val_X = train_df[dev_index,:], train_df[val_index,:]
                    dev_y, val_y = y[dev_index], y[val_index]
                    
                else:
                    dev_X, val_X = train_df.iloc[dev_index,:], train_df.iloc[val_index,:]
                    dev_y, val_y = y[dev_index], y[val_index]
                    print(dev_y.mean())  

                self.params['seed'] = nbagi*201706+231
                if custEval == False:
                    self.train(dev_X, dev_y, val_X, val_y, custEval = False)
                else:
                    self.train(dev_X, dev_y, val_X, val_y, custEval = True)
                preds = self.predict(val_X, predCol)

                oob_valpred[val_index,:] += preds
                if predCol == 1:
                    cv_r2.append(mean_absolute_error(val_y, preds))    
                    print(mean_absolute_error(val_y, preds))
                    print(cv_r2)
                    print(np.mean(cv_r2))
                predtst = self.predict(test_df, predCol)
                oob_tstpred += predtst
        oob_tstpred /= (nfold* nbag)
        oob_valpred /= (nbag) 
        return oob_tstpred, oob_valpred, cv_scores, cv_r2

    def create_feature_map(self, features):
        outfile = open('xgb.fmap', 'w')
        i = 0
        for feat in features:
            outfile.write('{0}\t{1}\tq\n'.format(i, feat))
            i = i + 1
        outfile.close()

    def genFeatureImportances(self, train_df):
        self.create_feature_map(train_df.columns)
        importance = self.model.get_fscore(fmap='xgb.fmap')
        importance = sorted(importance.items(), key=operator.itemgetter(1))

        df = pd.DataFrame(importance, columns=['feature', 'fscore'])
        df['fscore'] = df['fscore'] / df['fscore'].sum()

        plt.figure()
        df.plot()
        df.plot(kind='barh', x='feature', y='fscore', legend=False, figsize=(6, 10))
        plt.title('XGBoost Feature Importance')
        plt.xlabel('relative importance')
        plt.gcf().savefig('feature_importance_xgb.png')    

        df.sort_values(['fscore'],ascending=False, inplace=True)
        df.reset_index(drop=True)
        print(df)
        print(df['feature'][:10])
        return df
    def xgbCV(self, train, y, num_boost_round = 10000, nfold = 5, stratified=True,metrics={'mae'}):
        dtrain = xgb.DMatrix(train, y)
        xgb.cv(self.params, dtrain, num_boost_round=num_boost_round, nfold=nfold, stratified=stratified, metrics=metrics, obj=None, feval=None, maximize=False, early_stopping_rounds=200, fpreproc=None, as_pandas=True, verbose_eval=None, show_stdv=True, seed=12345, callbacks=[xgb.callback.print_evaluation(show_stdv=True)])
        
    def xgboostcv(self,max_depth,subsample,colsample_bytree,gamma,#n_estimators,#learning_rate,
                  min_child_weight,alpha,reg_lambda,colsample_bylevel,seed=201705,nthread=3,eval_metric="mae",
                  #scale_pos_weight=pos_wt[0],
                  objective='reg:linear',booster="gbtree"):

        num_rounds = 350#n_estimators
        params = {}
        params['objective'] = objective
        params['eta'] = 0.05
        params['max_depth'] = int(max_depth)
        params['silent'] = 1
        params['eval_metric'] = eval_metric
        params['min_child_weight'] = min_child_weight
        params['subsample'] = subsample
        params['colsample_bytree'] = colsample_bytree
        params['seed'] = seed
        params['nthread'] = nthread
        #params['scale_pos_weight'] = scale_pos_weight
        params['base_score'] = 0.011
        #params['tree_method'] = 'approx'#'exact'#/
        params['alpha'] = alpha
        params['lambda'] = reg_lambda
        params['gamma'] = gamma
        params['colsample_bylevel'] = colsample_bylevel
        self.params = params

        self.train(self.dev_X, self.dev_y, self.val_X, self.val_y, num_rounds = num_rounds)
        preds = self.predict(self.val_X)
        loss = mean_absolute_error(self.val_y, preds)
        print "SCORE:", loss
        return -1 * loss
	
    def getOptimParams(self, train_df, y, custSplit = False, trainx = None, valx = None, trainy = None, valy = None):
        xgboostBO = BayesianOptimization(self.xgboostcv,
								         {'max_depth': (int(5), int(8)),
								          #'learning_rate': (0.03, 0.03),
								          #'n_estimators': (int(200), int(2000)),
								          'subsample': (0.8, 1.0),
								          'colsample_bytree': (0.7, 1.0),
								          'gamma': (0.01, 1.0),
								          'min_child_weight': (0.01, 10.0),
								          'alpha': (0.01,5.0),
								          'reg_lambda': (0.01,5.0),
								          'colsample_bylevel': (0.8, 1.0) 
								         })

        if custSplit == True:
            self.dev_X, self.val_X = trainx, valx
            self.dev_y, self.val_y = trainy, valy
            self.y = y
        
        else:
            kf = model_selection.StratifiedKFold(n_splits = 5, shuffle=True, random_state = 12345)
            for dev_index, val_index in kf.split(train_df, y):
                self.dev_X, self.val_X = train_df.iloc[dev_index,:], train_df.iloc[val_index,:]
                self.dev_y, self.val_y = y[dev_index], y[val_index]
                self.y = y
        print("train Shape:",self.dev_X.shape,"    ", "val Shape:",self.val_X.shape)
        xgboostBO.maximize(init_points=20, n_iter=100, acq='ei')
        print('-'*53)

        print('Final Results')
        print('XGBOOST: %f' % xgboostBO.res['max']['max_val'])	
        print(xgboostBO.res['max'])


