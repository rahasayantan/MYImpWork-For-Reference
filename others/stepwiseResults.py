import pandas as pd
import numpy as np
import datetime as dt
import gc
from sklearn.preprocessing import MinMaxScaler
from encoding import cat2MeanShiftEncode
from sklearn.externals import joblib
from stepwiseSelectionRegression import stepwiseOLS

'''
dftrainStat = joblib.load("../input2/trainStat33.pkl")
dftrainStat2 = joblib.load("../input2/trainStat06.pkl")
dftrainStat3 = joblib.load("../input2/trainStat39.pkl")
'''
trainparcel = joblib.load("../input2/trainparcel.pkl")
train1 = joblib.load("../input/trainnum9Impute.pkl")
train2 = joblib.load("../input/trainlblcat.pkl")
#trainparcel = pd.DataFrame(trainparcel)
#trainparcel.columns = ['parcelid']
train = train#pd.concat((trainparcel,train1,train2), axis = 1)
train_y = joblib.load("../input2/y.pkl")
train['train_y'] = train_y
'''
train.set_index('parcelid', inplace = True)
dftrainStat.set_index('parcelid', inplace = True)
dftrainStat2.set_index('parcelid', inplace = True)
dftrainStat3.set_index('parcelid', inplace = True)

train = train.join(dftrainStat)
train = train.join(dftrainStat2, rsuffix='1')
train = train.join(dftrainStat3, rsuffix='2')
'''

stepwiseOLS(train, train_y)

#'taxdelinquencyyear_countenc0', 'poolcnt_countenc0', 'hashottuborspa_countenc0','bedroomcnt_countenc0',
#'regionidzip_meanshftenc0', 'assessmentyear_meanshftenc0', 'buildingclasstypeid_meanshftenc0', 'propertycountylandusecode_meanshftenc0', 'rawcensustractandblock_meanshftenc0', 'taxdelinquencyflag_meanshftenc0', 'propertyzoningdesc_meanshftenc0', 'poolcnt_meanshftenc0'
#'regionidzip_medianenc0', 'assessmentyear_medianenc0', 'buildingclasstypeid_medianenc0','propertycountylandusecode_medianenc0', 'rawcensustractandblock_medianenc0', 'taxdelinquencyflag_medianenc0', 'propertyzoningdesc_medianenc0', 'poolcnt_medianenc0'
#'regionidzip_meanenc0', 'assessmentyear_meanenc0', 'buildingclasstypeid_meanenc0','propertycountylandusecode_meanenc0', 'rawcensustractandblock_meanenc0', 'taxdelinquencyflag_meanenc0', 'propertyzoningdesc_meanenc0', 'poolcnt_meanenc0'




