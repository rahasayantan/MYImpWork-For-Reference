import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, log_loss
from statsmodels.formula.api import logit
import multiprocessing
import warnings
warnings.filterwarnings("ignore")

def evalBestCols(cols, train_df, val):
    formula = 'train_y ~ {} + 1'.format(cols)
    logitMod = logit(formula, train_df)
    res = logitMod.fit(method = "newton", disp = False)
    
    colLst = [x for x in list(res.pvalues[(res.pvalues <= val)].index) if x != 'Intercept']
    
    dropLst = pd.DataFrame({'colName': list(res.pvalues[(res.pvalues > val)].index),
                            'pvalue': list(res.pvalues[(res.pvalues > val)])
                            })
    dropLst.sort(['pvalue'], ascending = [1], inplace = True)
    toDropLst = list(dropLst.colName.values)
    return colLst, toDropLst
    
def computeScore(fieldLst, train_df, val):
    scores = []
    cols = '+'.join(fieldLst)
    candidate = fieldLst[-1]
    colLst, toDropLst = evalBestCols(cols, train_df, val)
    while (len(toDropLst) >= 2) and (not candidate in toDropLst):
        toDropLst = toDropLst[:-1]
        newLst = colLst + toDropLst 
        cols = '+'.join(newLst)
        colLst, toDropLst = evalBestCols(cols, train_df, val)
        
    if (not colLst) | (not candidate in colLst):
        scores.append(-1 * np.Inf, fieldlst)
    else:
        cols = '+'.join(colLst)
        formula = 'train_y ~ {} + 1'.format(cols)
        logitMod = logit(formula, train_df)
        res = logitMod.fit(method = "newton", disp = False)
        walChi = res.tvalues[[x for x in list(res.tvalues.index) if x == candidate]][0] ** 2
        scores.append(waldChi, colLst)
    return scores
    
class NoDaemonProcess(multiprocessing.Process):
    def _get_daemon(self):
        return False
    def _set_daemon(self):
        pass
    daemon = property(_get_daemon, _set_daemon)
    
class Pool(multiprocessing.pool.Pool):
    Process = NoDaemonProcess
    
def wrapperEvalCols(args):
    return getCols(*args)
    
def getCols(remaining, bestCand, candidate, train_df, val = 0.0001):
    fieldLst = bestCand + [candidate]
    result =  computeScore(fieldLst, train_df, val)
    return result
    
    
if __name__ == '__main__':
    train_df, train_y, test_df, test_y = getData(Normalize = False) ## This function need creation
    train_df['train_y'] = train_y
    iteration = 0
    
    remaining = set(train_df.columns) - set(['train_y'])       
    candidates = []
    bestCand = []
    baseLoss = -1 * np.Inf
    currScore, bestScore = baseLoss, baseLoss
    while remaining and currScore == bestScore:
        scores = []
        inputData = [(remaining, besCand, candidate, train_df, 0.0001) for candidate in remaining]
        pool = Pool(4)
        result = pool.map(wrapperEvalCols, inputData)
        for x in np.arange(len(result)):
            scores.append(result[x][0])
            
        scores.sort()
        newScore, newCand = scores.pop()
        
        if newScore != -1*np.Inf:
            for i in newCand:
                remaining.discard(i)
            bestCand = newcand
            bestScore = newScore
            currScore = bestScore
            candidates.append(bestCand)
        else:
            break
    print(bestCand)
