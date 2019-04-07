import pandas as pd
import numpy as np
import datetime as dt
import gc
from sklearn.preprocessing import MinMaxScaler
from encoding import cat2MeanShiftEncode
from sklearn.externals import joblib
from sklearn.metrics import roc_auc_score, log_loss
from statsmodels.formula.api import ols
import multiprocessing
import multiprocessing.pool

import warnings
warnings.filterwarnings("ignore")

def evalBestCols(cols, train2, val):
    formula = 'train_y ~ {} + 1'.format(cols)
    print(formula)
    olsMod = ols(formula=formula, data=train2)
    res = olsMod.fit()
    
    colLst = [x for x in list(res.pvalues[(res.pvalues <= val)].index) if x != 'Intercept']
    
    dropLst = pd.DataFrame({'colName': list(res.pvalues[(res.pvalues > val)].index),
                            'pvalue': list(res.pvalues[(res.pvalues > val)])
                            })
    dropLst.sort_values(['pvalue'], ascending = [1], inplace = True)
    toDropLst = list(dropLst.colName.values)
    return colLst, toDropLst
    
def computeScore(fieldLst, train2, val):
    scores = []
    cols = '+'.join(fieldLst)
    candidate = fieldLst[-1]
    colLst, toDropLst = evalBestCols(cols, train2, val)
    while (len(toDropLst) >= 2) and (not candidate in toDropLst):
        toDropLst = toDropLst[:-1]
        newLst = colLst + toDropLst 
        cols = '+'.join(newLst)
        colLst, toDropLst = evalBestCols(cols, train2, val)
        
    if (not colLst) | (not candidate in colLst):
        scores.append((-1 * np.Inf, fieldLst))
    else:
        cols = '+'.join(colLst)
        formula = 'train_y ~ {} + 1'.format(cols)
        olsMod = ols(formula=formula, data=train2)
        res = olsMod.fit()
        rsq = res.rsquared_adj
        scores.append((rsq, colLst))
    return scores
    
class NoDaemonProcess(multiprocessing.Process):
    def _get_daemon(self):
        return False
    def _set_daemon(self, value):
        pass
    daemon = property(_get_daemon, _set_daemon)
    
class Pool(multiprocessing.pool.Pool):
    Process = NoDaemonProcess
    
def wrapperEvalCols(args):
    return getCols(*args)
    
def getCols(remaining, bestCand, candidate, train2, val = 0.0001):
    fieldLst = bestCand + [candidate]
    result =  computeScore(fieldLst, train2, val)
    return result
    
    
def stepwiseOLS(train2, train_y):   
    remaining = set(train2.columns) - set(['train_y'])       
    candidates = []
    bestCand = []
    baseLoss = -1 * np.Inf
    currScore, bestScore = baseLoss, baseLoss
    while remaining and currScore == bestScore:
        scores = []
        inputData = [(remaining, bestCand, candidate, train2, 0.0001) for candidate in remaining]
        pool = Pool(4)
        result = pool.map(wrapperEvalCols, inputData)
        for x in np.arange(len(result)):
            scores.append(result[x][0])
            
        scores.sort()
        newScore, newCand = scores.pop()
        if newScore != -1*np.Inf:
            for i in newCand:
                remaining.discard(i)
            bestCand = newCand
            bestScore = newScore
            currScore = bestScore
            candidates.append(bestCand)
        else:
            break
    print(bestCand)
    
    
#fieldlst = ['regionidzip_meanshftenc0', 'calculatedfinishedsquarefeet', 'buildingclasstypeid_meanshftenc0', 'taxamount', 'propertycountylandusecode_meanshftenc0', 'taxvaluedollarcnt', 'propertyzoningdesc_meanshftenc0', 'rawcensustractandblock_meanshftenc0', 'taxdelinquencyflag_meanshftenc0', 'garagetotalsqft', 'countnullcol', 'regionidneighborhood_meanshftenc0']
    
