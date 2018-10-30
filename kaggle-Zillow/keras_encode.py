###XGB Finale - No Outliers
from sklearn import model_selection, preprocessing, ensemble
import pandas as pd
import numpy as np
import datetime as dt
import gc
from sklearn.metrics import mean_absolute_error
from sklearn.externals import joblib
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers.advanced_activations import PReLU
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers.normalization import BatchNormalization
from keras.layers.core import Dense, Dropout, Activation, Reshape
from keras.layers.embeddings import Embedding
from keras.layers import Merge
from keras.optimizers import Adam

train = joblib.load("../input/trainstat.pkl")

cols = [a for a in [y for y in [x for x in [a for a in train.columns if 'X' not in a] if 'countenc0' not in x] if 'meanshftenc0' not in y] if 'yrmonth' not in a]

cols = cols +['X4_Mon_logerror3std',
'X3_Mon_logerror6skew',
'X3_Mon_logerror3mean',
'X4_Mon_logerror3skew',
'X3_Mon_logerror3std',
'X1_Mon_logerror6skew',
'X1_Mon_logerror3std',
'X1_Mon_logerror3skew',
'X3_Mon_logerror3skew',
'X2_Mon_logerror3skew',
'X4_Mon_logerror6skew',
'X4_Mon_logerror6std',
'X4_Mon_logerror3mean',
'X1_Mon_logerror6mean']

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
yval = y[valindex]
#train = train[trainindex]#
#y = y[trainindex]
#################################################

lbound = np.mean(y) - 3 * np.std(y)
ubound = np.mean(y) + 3 * np.std(y)

test = joblib.load("../input/teststat.pkl")
test = test[cols]
#test = valid.copy()

gc.collect()

all_cols = cols

catcols = [u'yearbuilt', 
       u'bathroomcnt', u'bedroomcnt', u'buildingqualitytypeid',
       u'buildingclasstypeid', u'calculatedbathnbr', 
       u'threequarterbathnbr', u'fips', u'fireplacecnt',
       u'fullbathcnt', u'garagecarcnt',
       u'heatingorsystemtypeid', u'numberofstories', u'poolcnt',
       u'propertycountylandusecode', u'propertylandusetypeid',
       u'propertyzoningdesc', u'rawcensustractandblock',
       u'censustractandblock', u'regionidcounty', u'regionidcity',
       u'regionidzip', u'regionidneighborhood',
       u'typeconstructiontypeid', u'unitcnt', u'yardbuildingsqft17',
       u'yardbuildingsqft26', u'assessmentyear',
       u'taxdelinquencyyear', u'roomcnt', u'latitudetrim', u'longitudetrim',
       u'airconditioningtypeid', u'architecturalstyletypeid', u'yr', u'month',
       u'qtr']
numcols = list(set(all_cols) - set(catcols))

train = pd.concat((train[catcols], train[numcols]), axis = 1)
test = pd.concat((test[catcols], test[numcols]), axis = 1)
valid = pd.concat((valid[catcols], valid[numcols]), axis = 1)

train_test = pd.concat((train, test),axis =0)
gc.collect()

from sklearn.preprocessing import LabelEncoder, StandardScaler

for col in catcols:
    enc = LabelEncoder()
    enc.fit(train_test[col])
    train[col] = enc.transform(train[col])
    test[col] = enc.transform(test[col])   
    train_test[col] = enc.transform(train_test[col]) 
    valid[col] = enc.transform(valid[col]) 

for col in numcols:
    enc = preprocessing.StandardScaler()#MinMaxScaler()
    enc.fit(train_test[col])
    train[col] = enc.transform(train[col])
    test[col] = enc.transform(test[col])   
    train_test[col] = enc.transform(train_test[col]) 
    valid[col] = enc.transform(valid[col]) 
    
del(train_test); gc.collect()
       
def splitdata(X):
    xlst =[]
    xlst.append(X[...,[0]])
    xlst.append(X[...,[1]])
    xlst.append(X[...,[2]])
    xlst.append(X[...,[3]])
    xlst.append(X[...,[4]])
    xlst.append(X[...,[5]])
    xlst.append(X[...,[6]])
    xlst.append(X[...,[7]])
    xlst.append(X[...,[8]])
    xlst.append(X[...,[9]])
    xlst.append(X[...,[10]])
    xlst.append(X[...,[11]])
    xlst.append(X[...,[12]])
    xlst.append(X[...,[13]])
    xlst.append(X[...,[14]])
    xlst.append(X[...,[15]])
    xlst.append(X[...,[16]])
    xlst.append(X[...,[17]])
    xlst.append(X[...,[18]])
    xlst.append(X[...,[19]])
    xlst.append(X[...,[20]])
    xlst.append(X[...,[21]])
    xlst.append(X[...,[22]])
    xlst.append(X[...,[23]])
    xlst.append(X[...,[24]])
    xlst.append(X[...,[25]])
    xlst.append(X[...,[26]])
    xlst.append(X[...,[27]])
    xlst.append(X[...,[28]])
    xlst.append(X[...,[29]])
    xlst.append(X[...,[30]])
    xlst.append(X[...,[31]])
    xlst.append(X[...,[32]])
    xlst.append(X[...,[33]])
    xlst.append(X[...,[34]])
    xlst.append(X[...,[35]])
    xlst.append(X[...,[36]])
    xlst.append(X[...,[37]])
    xlst.append(X[...,[38]])
    xlst.append(X[...,[39]])
    xlst.append(X[...,[40]])
    xlst.append(X[...,[41]])
    xlst.append(X[...,[42]])
    xlst.append(X[...,[43]])
    xlst.append(X[...,[44]])
    xlst.append(X[...,[45]])
    xlst.append(X[...,[46]])
    xlst.append(X[...,[47]])
    xlst.append(X[...,[48]])
    xlst.append(X[...,[49]])
    xlst.append(X[...,[50]])
    xlst.append(X[...,[51]])
    xlst.append(X[...,[52]])
    xlst.append(X[...,[53]])
    xlst.append(X[...,[54]])
    xlst.append(X[...,[55]])
    xlst.append(X[...,[56]])
    xlst.append(X[...,[57]])
    xlst.append(X[...,[58]])
    xlst.append(X[...,[59]])
    xlst.append(X[...,[60]])
    xlst.append(X[...,[61]])
    xlst.append(X[...,[62]])
    xlst.append(X[...,[63]])
    xlst.append(X[...,[64]])
    xlst.append(X[...,[65]])
    xlst.append(X[...,[66]])
    xlst.append(X[...,[67]])
    xlst.append(X[...,[68]])
    xlst.append(X[...,[69]])

    xlst.append(X[...,[70]])
    xlst.append(X[...,[71]])
    xlst.append(X[...,[72]])
    xlst.append(X[...,[73]])
    xlst.append(X[...,[74]])
    xlst.append(X[...,[75]])
    xlst.append(X[...,[76]])
    xlst.append(X[...,[77]])
    xlst.append(X[...,[78]])
    xlst.append(X[...,[79]])
    xlst.append(X[...,[80]])
    xlst.append(X[...,[81]])
    xlst.append(X[...,[82]])
    xlst.append(X[...,[83]])
    return xlst

def nn_model():
    models = []
    
    x0 = Sequential()
    x0.add(Embedding(184, 30, input_length=1))
    x0.add(Reshape(target_shape=(30,)))
    models.append(x0)
    
    x1 = Sequential()
    x1.add(Embedding(40, 6, input_length=1))
    x1.add(Reshape(target_shape=(6,)))
    models.append(x1)
    
    x2 = Sequential()
    x2.add(Embedding(26, 4, input_length=1))
    x2.add(Reshape(target_shape=(4,)))
    models.append(x2)    
    
    x3 = Sequential()
    x3.add(Embedding(13, 2, input_length=1))
    x3.add(Reshape(target_shape=(2,)))
    models.append(x3)    
    
    x4 = Sequential()
    x4.add(Embedding(6, 1, input_length=1))
    x4.add(Reshape(target_shape=(1,)))
    models.append(x4)    
    
    x5 = Sequential()
    x5.add(Embedding(37, 6, input_length=1))
    x5.add(Reshape(target_shape=(6,)))
    models.append(x5)
    
    x6 = Sequential()
    x6.add(Embedding(8, 1, input_length=1))
    x6.add(Reshape(target_shape=(1,)))
    models.append(x6)    
    
    x7 = Sequential()
    x7.add(Embedding(4, 1, input_length=1))
    x7.add(Reshape(target_shape=(1,)))
    models.append(x7)    
    
    x8 = Sequential()
    x8.add(Embedding(10, 2, input_length=1))
    x8.add(Reshape(target_shape=(2,)))
    models.append(x8)  
    
    x9 = Sequential()
    x9.add(Embedding(23, 4, input_length=1))
    x9.add(Reshape(target_shape=(4,)))
    models.append(x9)      
    
    x10 = Sequential()
    x10.add(Embedding(25, 5, input_length=1))
    x10.add(Reshape(target_shape=(5,)))
    models.append(x10)    
    
    x11 = Sequential()
    x11.add(Embedding(15, 3, input_length=1))
    x11.add(Reshape(target_shape=(3,)))
    models.append(x11)    
    
    x12 = Sequential()
    x12.add(Embedding(13, 2, input_length=1))
    x12.add(Reshape(target_shape=(2,)))
    models.append(x12)    
    
    x13 = Sequential()
    x13.add(Embedding(2, 1, input_length=1))
    x13.add(Reshape(target_shape=(1,)))
    models.append(x13)    
    
    x14 = Sequential()
    x14.add(Embedding(235, 50, input_length=1))
    x14.add(Reshape(target_shape=(50,)))
    models.append(x14)    
    
    x15 = Sequential()
    x15.add(Embedding(17, 3, input_length=1))
    x15.add(Reshape(target_shape=(3,)))
    models.append(x15)    

    x16 = Sequential()
    x16.add(Embedding(5652, 50, input_length=1))
    x16.add(Reshape(target_shape=(50,)))
    models.append(x16)    
    
    x17 = Sequential()
    x17.add(Embedding(472, 50, input_length=1))
    x17.add(Reshape(target_shape=(50,)))
    models.append(x17)    
    
    x18 = Sequential()
    x18.add(Embedding(458, 50, input_length=1))
    x18.add(Reshape(target_shape=(50,)))
    models.append(x18)    

    x19 = Sequential()
    x19.add(Embedding(4, 1, input_length=1))
    x19.add(Reshape(target_shape=(1,)))
    models.append(x19)    
    
    x20 = Sequential()
    x20.add(Embedding(187, 30, input_length=1))
    x20.add(Reshape(target_shape=(30,)))
    models.append(x20)    
    
    x21 = Sequential()
    x21.add(Embedding(404, 50, input_length=1))
    x21.add(Reshape(target_shape=(50,)))
    models.append(x21)
    
    x22 = Sequential()
    x22.add(Embedding(530, 50, input_length=1))
    x22.add(Reshape(target_shape=(50,)))
    models.append(x22)                
    
    x23 = Sequential()
    x23.add(Embedding(6, 1, input_length=1))
    x23.add(Reshape(target_shape=(1,)))
    models.append(x23)        
    
    x24 = Sequential()
    x24.add(Embedding(155, 30, input_length=1))
    x24.add(Reshape(target_shape=(30,)))
    models.append(x24)  
    
    x25 = Sequential()
    x25.add(Embedding(1665, 50, input_length=1))
    x25.add(Reshape(target_shape=(50,)))
    models.append(x25)              
    
    x26 = Sequential()
    x26.add(Embedding(595, 50, input_length=1))
    x26.add(Reshape(target_shape=(50,)))
    models.append(x26)        
    
    x27 = Sequential()
    x27.add(Embedding(15, 3, input_length=1))
    x27.add(Reshape(target_shape=(3,)))
    models.append(x27)        
    
    x28 = Sequential()
    x28.add(Embedding(32, 6, input_length=1))
    x28.add(Reshape(target_shape=(6,)))
    models.append(x28)        
    
    x29 = Sequential()
    x29.add(Embedding(37, 6, input_length=1))
    x29.add(Reshape(target_shape=(6,)))
    models.append(x29)  
    
    x30 = Sequential()
    x30.add(Embedding(149, 30, input_length=1))
    x30.add(Reshape(target_shape=(30,)))
    models.append(x30)              
    
    x31 = Sequential()
    x31.add(Embedding(190, 30, input_length=1))
    x31.add(Reshape(target_shape=(30,)))
    models.append(x31) 
    
    x32 = Sequential()
    x32.add(Embedding(8, 2, input_length=1))
    x32.add(Reshape(target_shape=(2,)))
    models.append(x32) 

    x33 = Sequential()
    x33.add(Embedding(9, 2, input_length=1))
    x33.add(Reshape(target_shape=(2,)))
    models.append(x33)                  
    
    x34 = Sequential()
    x34.add(Embedding(2, 1, input_length=1))
    x34.add(Reshape(target_shape=(1,)))
    models.append(x34)                  
    
    x35 = Sequential()
    x35.add(Embedding(12, 2, input_length=1))
    x35.add(Reshape(target_shape=(2,)))
    models.append(x35)                  
    
    x36 = Sequential()
    x36.add(Embedding(4, 1, input_length=1))
    x36.add(Reshape(target_shape=(1,)))
    models.append(x36)                  
    
    x37 = Sequential()
    x37.add(Dense(1, input_dim=1))
    models.append(x37)    
    
    x38 = Sequential()
    x38.add(Dense(1, input_dim=1))
    models.append(x38)    
    
    x39 = Sequential()
    x39.add(Dense(1, input_dim=1))
    models.append(x39)            
    
    x40 = Sequential()
    x40.add(Dense(1, input_dim=1))
    models.append(x40)
    
    x41 = Sequential()
    x41.add(Dense(1, input_dim=1))
    models.append(x41)
    
    x42 = Sequential()
    x42.add(Dense(1, input_dim=1))
    models.append(x42)

    x43 = Sequential()
    x43.add(Dense(1, input_dim=1))
    models.append(x43)

    x44 = Sequential()
    x44.add(Dense(1, input_dim=1))
    models.append(x44)

    x45 = Sequential()
    x45.add(Dense(1, input_dim=1))
    models.append(x45)

    x46 = Sequential()
    x46.add(Dense(1, input_dim=1))
    models.append(x46)

    x47 = Sequential()
    x47.add(Dense(1, input_dim=1))
    models.append(x47)

    x48 = Sequential()
    x48.add(Dense(1, input_dim=1))
    models.append(x48)

    x49 = Sequential()
    x49.add(Dense(1, input_dim=1))
    models.append(x49)
                    
    x50 = Sequential()
    x50.add(Dense(1, input_dim=1))
    models.append(x50)

    x51 = Sequential()
    x51.add(Dense(1, input_dim=1))
    models.append(x51)

    x52 = Sequential()
    x52.add(Dense(1, input_dim=1))
    models.append(x52)

    x53 = Sequential()
    x53.add(Dense(1, input_dim=1))
    models.append(x53)

    x54 = Sequential()
    x54.add(Dense(1, input_dim=1))
    models.append(x54)

    x55 = Sequential()
    x55.add(Dense(1, input_dim=1))
    models.append(x55)

    x56 = Sequential()
    x56.add(Dense(1, input_dim=1))
    models.append(x56)

    x57 = Sequential()
    x57.add(Dense(1, input_dim=1))
    models.append(x57)

    x58 = Sequential()
    x58.add(Dense(1, input_dim=1))
    models.append(x58)

    x59 = Sequential()
    x59.add(Dense(1, input_dim=1))
    models.append(x59)

    x60 = Sequential()
    x60.add(Dense(1, input_dim=1))
    models.append(x60)

    x61 = Sequential()
    x61.add(Dense(1, input_dim=1))
    models.append(x61)

    x62 = Sequential()
    x62.add(Dense(1, input_dim=1))
    models.append(x62)
    
    x63 = Sequential()
    x63.add(Dense(1, input_dim=1))
    models.append(x63)
    
    x64 = Sequential()
    x64.add(Dense(1, input_dim=1))
    models.append(x64)
    
    x65 = Sequential()
    x65.add(Dense(1, input_dim=1))
    models.append(x65)
    
    x66 = Sequential()
    x66.add(Dense(1, input_dim=1))
    models.append(x66)
    
    x67 = Sequential()
    x67.add(Dense(1, input_dim=1))
    models.append(x67)
    
    x68 = Sequential()
    x68.add(Dense(1, input_dim=1))
    models.append(x68)
    
    x69 = Sequential()
    x69.add(Dense(1, input_dim=1))
    models.append(x69)

    x70 = Sequential()
    x70.add(Dense(1, input_dim=1))
    models.append(x70)

    x71 = Sequential()
    x71.add(Dense(1, input_dim=1))
    models.append(x71)
    x72 = Sequential()
    x72.add(Dense(1, input_dim=1))
    models.append(x72)
    x73 = Sequential()
    x73.add(Dense(1, input_dim=1))
    models.append(x73)
    x74 = Sequential()
    x74.add(Dense(1, input_dim=1))
    models.append(x74)
    x75 = Sequential()
    x75.add(Dense(1, input_dim=1))
    models.append(x75)
    x76 = Sequential()
    x76.add(Dense(1, input_dim=1))
    models.append(x76)
    x77 = Sequential()
    x77.add(Dense(1, input_dim=1))
    models.append(x77)
    x78 = Sequential()
    x78.add(Dense(1, input_dim=1))
    models.append(x78)
    x79 = Sequential()
    x79.add(Dense(1, input_dim=1))
    models.append(x79)

    x80 = Sequential()
    x80.add(Dense(1, input_dim=1))
    models.append(x80)
    x81 = Sequential()
    x81.add(Dense(1, input_dim=1))
    models.append(x81)
    x82 = Sequential()
    x82.add(Dense(1, input_dim=1))
    models.append(x82)
    x83 = Sequential()
    x83.add(Dense(1, input_dim=1))
    models.append(x83)

    model = Sequential()
    model.add(Merge(models, mode='concat'))
    model.add(Dropout(0.2))
    model.add(Dense(800, init = 'uniform'))#500
    model.add(PReLU())
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(Dense(800, init = 'uniform'))#400
    model.add(PReLU())
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(Dense(500, init = 'uniform'))#400
    model.add(PReLU())
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    model.add(Dense(500, init = 'uniform'))#400
    model.add(PReLU())
    model.add(BatchNormalization())
    model.add(Dropout(0.1))
    model.add(Dense(1, init='zero'))
    model.compile(loss = 'mean_absolute_error', optimizer = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.))
    return(model)    

oobval = np.zeros((train.shape[0],1))
oobtest = np.zeros((test.shape[0],1))
valerr = []
val_scores = []
cnt = 0   
nbag = 1
nfold = 5
np.random.seed(2017)
for x in np.arange(nbag):
    for seed in [2017]:
        kf = model_selection.KFold(n_splits=nfold, shuffle=True, random_state=seed)    
        for dev_index, val_index in kf.split(y):
            dev_X, val_X = train.values[dev_index], train.values[val_index]
            dev_y, val_y = y[dev_index], y[val_index]
            #dev_X = dev_X[(dev_y > lbound) & (dev_y < ubound)]
            #dev_y = dev_y[(dev_y > lbound) & (dev_y < ubound)]
            #val_X2 = val_X[(val_y > lbound) & (val_y < ubound)]
            #val_y2 = val_y[(val_y > lbound) & (val_y < ubound)]
            
            model = nn_model()
            earlyStopping=EarlyStopping(monitor='val_loss', patience=20, verbose=1, mode='auto')
            checkpointer = ModelCheckpoint(filepath="./weights2XXLK.hdf5", verbose=1, save_best_only=True)
            fit = model.fit(splitdata(dev_X),dev_y, batch_size=512,nb_epoch = 100,validation_data=(splitdata(val_X), val_y),verbose = 1, callbacks=[earlyStopping,checkpointer])
            print("loading weights")
            model.load_weights("./weights2XXLK.hdf5")
            print("predicting..")
            preds = model.predict(splitdata(val_X))
            oobval[val_index] += preds
            oobtest += model.predict(splitdata(test.values)).reshape(-1,1)
            valerr.append(mean_absolute_error(val_y, preds))
            print(valerr, "mean:", np.mean(valerr), "std:", np.std(valerr))
            val_scores.append(mean_absolute_error(model.predict(splitdata(valid.values)), yval))
            print(val_scores, np.mean(val_scores),"---", np.std(val_scores))            
            
pred2 = oobtest / (nbag * nfold)
oobpred2 = oobval / (nbag )
print(mean_absolute_error(y, oobpred2))                    

joblib.dump(oobpred2,'../input/train_keras_enc_stat.pkl')
joblib.dump(pred2,'../input/test_keras_enc_stat.pkl')

