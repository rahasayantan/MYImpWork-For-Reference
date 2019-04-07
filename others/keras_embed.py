print ('Good luck')
from sklearn import model_selection, preprocessing, ensemble
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
import os
os.environ['OMP_NUM_THREADS'] = '3'
import gc
from sklearn import preprocessing
from sklearn.externals import joblib


train = joblib.load('../input/4/train_cleaned_smallhr_ds.pkl')
test = joblib.load('../input/4/test_cleaned_smallhr_ds.pkl')
y = joblib.load('../input/4/y_cleaned_smallhr_ds.pkl')
click_id = joblib.load('../input/4/click_id_cleaned_smallhr_ds.pkl')
gc.collect()
assert(set(train.columns) - set(test.columns) == set())
print("Read Complete..")

cols2enc = ['app', 'channel', 'ip', 'os', 'hour']
le = preprocessing.LabelEncoder() 
for vcol in cols2enc:
    le.fit(np.vstack((train[vcol].reshape(-1,1), test[vcol].reshape(-1,1))))
    train[vcol] = le.transform(train[vcol].reshape(-1,1))
    test[vcol] = le.transform(test[vcol].reshape(-1,1))

max_app = np.max([train['app'].max(), test['app'].max()])+1
max_ch = np.max([train['channel'].max(), test['channel'].max()])+1
max_ip = np.max([train['ip'].max(), test['ip'].max()])+1
max_os = np.max([train['os'].max(), test['os'].max()])+1
max_h = np.max([train['hour'].max(), test['hour'].max()])+1

cols2scale = ['mnenc_ip', 'mnenc_app',
       'mnenc_device', 'mnenc_os', 'mnenc_channel', 'cntenc_ip', 'cntenc_app',
       'cntenc_device', 'cntenc_channel', 'y_len1', 'y_len2', 'y_len3',
       'y_len4', 'y_len12', 'y_len42', 'y_mean61', 'y_mean62', 'y_mean63',
       'y_mean64']

le = preprocessing.MinMaxScaler() 
for vcol in cols2scale:
    le.fit(np.vstack((train[vcol].reshape(-1,1), test[vcol].reshape(-1,1))))
    train[vcol] = le.transform(train[vcol].reshape(-1,1))
    test[vcol] = le.transform(test[vcol].reshape(-1,1))
    if train[vcol].dtype == 'float64':
        train[vcol] = train[vcol].astype('float32')
        test[vcol] = test[vcol].astype('float32')

print ('neural network....')
import plaidml.keras
plaidml.keras.install_backend()
from keras.layers import Input, Embedding, Dense, Flatten, Dropout, concatenate, Reshape
from keras.layers import BatchNormalization, SpatialDropout1D, Conv1D
from keras.callbacks import Callback
from keras.models import Model
from keras.optimizers import Adam

def get_keras_data(dataset):
    X = {
        'app': np.array(dataset.app),
        'ch': np.array(dataset.channel),
        'ip': np.array(dataset.ip),
        'os': np.array(dataset.os),
        'hour': np.array(dataset.hour),
        'mnenc_ip': np.array(dataset.mnenc_ip),
        'mnenc_app': np.array(dataset.mnenc_app),
        'mnenc_device': np.array(dataset.mnenc_device),
        'mnenc_os': np.array(dataset.mnenc_os),
        'mnenc_channel': np.array(dataset.mnenc_channel),
        'cntenc_ip': np.array(dataset.cntenc_ip),
        'cntenc_app': np.array(dataset.cntenc_app),
        'cntenc_device': np.array(dataset.cntenc_device),
        'cntenc_channel': np.array(dataset.cntenc_channel),
        'y_len1': np.array(dataset.y_len1),
        'y_len2': np.array(dataset.y_len2),
        'y_len3': np.array(dataset.y_len3),
        'y_len4': np.array(dataset.y_len4),
        'y_len12': np.array(dataset.y_len12),
        'y_len42': np.array(dataset.y_len42),    
        'y_mean61': np.array(dataset.y_mean61),
        'y_mean62': np.array(dataset.y_mean62),
        'y_mean63': np.array(dataset.y_mean63),
        'y_mean64': np.array(dataset.y_mean64),
    }
    return X

from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, LearningRateScheduler, Callback

def nn():
    in_app = Input(shape=[1], name = 'app')
    emb_app = Embedding(max_app, 10)(in_app)
    emb_app = Reshape((10,))(emb_app)
    
    in_ch = Input(shape=[1], name = 'ch')
    emb_ch = Embedding(max_ch, 10)(in_ch)
    emb_ch = Reshape((10,))(emb_ch)
    
    in_ip = Input(shape=[1], name = 'ip')
    emb_ip = Embedding(max_ip, 50)(in_ip)
    emb_ip = Reshape((50,))(emb_ip)
        
    in_os = Input(shape=[1], name = 'os')
    emb_os = Embedding(max_os, 10)(in_os)
    emb_os = Reshape((10,))(emb_os)
        
    in_hr = Input(shape=[1], name = 'hour')
    emb_hr = Embedding(max_os, 2)(in_hr)
    emb_hr = Reshape((2,))(emb_hr)
    
    in_1 = Input(shape=[1], name = 'mnenc_ip')
    emb_1 = Dense(1)(in_1) 
    in_2 = Input(shape=[1], name = 'mnenc_app')
    emb_2 = Dense(1)(in_2) 
    in_3 = Input(shape=[1], name = 'mnenc_device')
    emb_3 = Dense(1)(in_3) 
    in_4 = Input(shape=[1], name = 'mnenc_os')
    emb_4 = Dense(1)(in_4) 
    in_5 = Input(shape=[1], name = 'mnenc_channel')
    emb_5 = Dense(1)(in_5) 
    in_6 = Input(shape=[1], name = 'cntenc_ip')
    emb_6 = Dense(1)(in_6)  
    in_7 = Input(shape=[1], name = 'cntenc_app')
    emb_7 = Dense(1)(in_7)  
    in_8 = Input(shape=[1], name = 'cntenc_device')
    emb_8 = Dense(1)(in_8) 
    in_9 = Input(shape=[1], name = 'cntenc_channel')
    emb_9 = Dense(1)(in_9) 
    in_10 = Input(shape=[1], name = 'y_len1')
    emb_10 = Dense(1)(in_10) 
    in_11 = Input(shape=[1], name = 'y_len2')
    emb_11 = Dense(1)(in_11)  
    in_12 = Input(shape=[1], name = 'y_len3')
    emb_12 = Dense(1)(in_12)             
    in_13 = Input(shape=[1], name = 'y_len4')
    emb_13 = Dense(1)(in_13) 
    in_14 = Input(shape=[1], name = 'y_len12')
    emb_14 = Dense(1)(in_14)  
    in_15 = Input(shape=[1], name = 'y_len42')
    emb_15 = Dense(1)(in_15) 
    in_16 = Input(shape=[1], name = 'y_mean61')
    emb_16 = Dense(1)(in_16) 
    in_17 = Input(shape=[1], name = 'y_mean62')
    emb_17 = Dense(1)(in_17)  
    in_18 = Input(shape=[1], name = 'y_mean63')
    emb_18 = Dense(1)(in_18) 
    in_19 = Input(shape=[1], name = 'y_mean64')
    emb_19 = Dense(1)(in_18)  

    fe = concatenate([(emb_app), (emb_ch), (emb_ip), (emb_os), (emb_hr), 
                     (emb_1), (emb_2), (emb_3), (emb_4), (emb_5),
                     (emb_6), (emb_7), (emb_8), (emb_9), (emb_10), 
                     (emb_11), (emb_12), (emb_13), (emb_14), (emb_15),
                     (emb_16), (emb_17), (emb_18), (emb_19)                                                              
                     ])

    s_dout = Dropout(0.2)(fe)#SpatialDropout1D(0.2)(fe)

    #fl1 = Flatten()(s_dout)

    #conv3 = Conv1D(8, kernel_size=3, strides=1, padding='same')(s_dout)
    #fl3 = Flatten()(conv3)

    #conv4 = Conv1D(16, kernel_size=4, strides=1, padding='same')(s_dout)
    #fl4 = Flatten()(conv4)

    #conv5 = Conv1D(32, kernel_size=5, strides=1, padding='same')(s_dout)
    #fl5 = Flatten()(conv5)

#    concat = concatenate([(fl1), (fl3), (fl4), (fl5)])

    x = Dropout(0.2)(Dense(100,activation='relu')(s_dout))#(concat))
    x = Dropout(0.2)(Dense(100,activation='relu')(x))
    x = Dropout(0.2)(Dense(50,activation='relu')(x))

    outp = Dense(1,activation='sigmoid')(x)

    model = Model(inputs=[in_app,in_ch,in_ip,in_os,in_hr,
                           in_1, in_2, in_3, in_4, in_5,
                           in_6, in_7, in_8, in_9, in_10,
                           in_11, in_12, in_13, in_14, in_15,
                           in_16, in_17, in_18, in_19], outputs=outp)
    return model
    
test = get_keras_data(test)

batch_size = 128
exp_decay = lambda init, fin, steps: (init/fin)**(1/(steps-1)) - 1
steps = int(len(train) / batch_size) * num_epoch
lr_init, lr_fin = 0.001, 0.0001
lr_decay = exp_decay(lr_init, lr_fin, steps)

oobtest = np.zeros((click_id.shape[0],1))
oobval = np.zeros((train.shape[0],1))
valerr = []
cnt = 0
#val_scores = []
cv_r2 = []
nfold = 3
nbag = 1
num_epoch = 5
class_weight = {0:.01,1:.99} # magic    
for i in np.arange(nbag): 
    for seed in [2018]:
        kf = model_selection.StratifiedKFold(n_splits= nfold, shuffle=True, random_state=seed*nbag)
        for dev_index, val_index in kf.split(oobval, y):
            dev_X, val_X = train.iloc[dev_index,:], train.iloc[val_index,:]
            dev_y, val_y = y[dev_index], y[val_index]
            print(dev_X.shape)  
            model = nn()
            earlystopper = EarlyStopping(monitor='val_loss', patience=20, verbose=1 , mode='auto')
            plateau = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, verbose=0, mode='auto', epsilon=0.0001, cooldown=0, min_lr=0)
            save_model_path = "nn_sav_"+str(i) +".h5"
            checkpointer = ModelCheckpoint(save_model_path, verbose=1, save_best_only=True)
            optimizer_adam = Adam(lr=0.001, decay=lr_decay)
            model.compile(loss='binary_crossentropy',optimizer=optimizer_adam,metrics=['accuracy'])
            gc.collect()
            dev_X = get_keras_data(dev_X)
            val_X = get_keras_data(val_X)
            print("Start Training")
            model.fit(dev_X,dev_y,batch_size=batch_size,
                        epochs=num_epoch,
                        verbose=1,
                        class_weight=class_weight, 
                        callbacks=[earlystopper, checkpointer, plateau],
                        validation_data=(val_X, val_y)
                     )
            pred = model.predict(val_X).reshape(-1,1)
            oobval[val_index,:] += pred
            cv_r2.append(roc_auc_score(val_y, pred))    
            print(cv_r2, np.mean(cv_r2),"---", np.std(cv_r2))
            oobtest += model.predict(test).reshape(-1,1)
            del(dev_X, dev_y, val_X, val_y, model)
            gc.collect()
            pred = oobtest / (nfold * nbag)
            oobpred = oobval / nbag
joblib.dump(oobpred,'../input/4/train_keras_embed_red_ds_5f.pkl')
joblib.dump(pred,'../input/4/test_keras_embed_red_ds_5f.pkl')


# Submission
# need to comment next 2 lines
#pred = oobtest
#oobpred = oobval

sub = pd.DataFrame()

test_click_id = click_id
sub['click_id'] = test_click_id.astype('int')
sub['is_attributed'] = pred

sub.to_csv('../output/4/sub_embed_keras.csv.gz',compression='gzip', index=False , float_format='%.5g')

