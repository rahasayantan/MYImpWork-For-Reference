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
from scipy.special import erfinv
from sklearn.metrics import roc_auc_score

train = joblib.load("../input/raw_train.pkl"
test = joblib.load("../input/raw_test.pkl")

### Drop unwanted columns
#Label encode all the categoricals
cols2enc = ['app', 'channel', 'ip', 'hour' xxxx]
le = preprocessing.LabelEncoder() 
for vcol in cols2enc:
    le.fit(np.vstack((train[vcol].reshape(-1,1), test[vcol].reshape(-1,1))))
    train[vcol] = le.transform(train[vcol].reshape(-1,1))
    test[vcol] = le.transform(test[vcol].reshape(-1,1))


max_app = np.max([train['app'].max(), test['app'].max()])+1
max_ch = np.max([train['channel'].max(), test['channel'].max()])+1
max_ip = np.max([train['ip'].max(), test['ip'].max()])+1
max_h = np.max([train['hour'].max(), test['hour'].max()])+1

y = xxx
click_id = xxx

def getTimeFeatures(df_train):
    df_train['day'] = df_train['click_time'].dt.day.astype('uint8').values
    df_train['hour'] = df_train['click_time'].dt.hour.astype('uint8').values
    df_train['min'] = df_train['click_time'].dt.minute.astype('uint8').values
    df_train['sec'] = df_train['click_time'].dt.second.astype('uint8').values
    #df_train.drop(['click_time'], axis = 1, inplace = True)
    return df_train

test = getTimeFeatures(test)
train = getTimeFeatures(train)

#### get some count features only



print ('neural network....')
#import plaidml.keras
#plaidml.keras.install_backend()
from keras.layers import Input, Embedding, Dense, Flatten, Dropout, concatenate, Reshape, Concatenate, BatchNormalization
from keras.layers import BatchNormalization, SpatialDropout1D, Conv1D, Activation
from keras.callbacks import Callback
from keras.models import Model
from keras.optimizers import Adam
from keras.models import Model, load_model
from  keras.regularizers import l2
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, LearningRateScheduler, Callback
from keras import backend as K

def get_keras_data(dataset1, y, batch_size = 64, shuffle=False):
    batch_index = 0
    n = dataset1.shape[0]
    while True:
        if batch_index == 0:
            index_array = np.arange(n)
            if shuffle:
                index_array = np.random.permutation(n)
        current_index = (batch_index * batch_size) % n
        if n >= current_index + batch_size:
            current_batch_size = batch_size
            batch_index += 1
        else:
            current_batch_size = n - current_index
            batch_index = 0
        batch_x = dataset1.iloc[current_index: current_index + current_batch_size,:]
        batch_y = y[current_index: current_index + current_batch_size]
        
        X = {
            'app': batch_x.app.values,
            'ch': batch_x.channel.values,
            'ip': batch_x.ip.values,
            'hour': batch_x.hour.values,
            xxxx
        }
        
        Y = { 
            'y' : batch_y
            }
        yield X, Y

def get_keras_data_T(dataset1, batch_size = 64, shuffle=False):
    batch_index = 0
    n = dataset1.shape[0]
    while True:
        if batch_index == 0:
            index_array = np.arange(n)
            if shuffle:
                index_array = np.random.permutation(n)
        current_index = (batch_index * batch_size) % n
        if n >= current_index + batch_size:
            current_batch_size = batch_size
            batch_index += 1
        else:
            current_batch_size = n - current_index
            batch_index = 0
        batch_x = dataset1.iloc[current_index: current_index + current_batch_size,:]
        batch_y = y[current_index: current_index + current_batch_size]
        
        X = {
            'app': batch_x.app.values,
            'ch': batch_x.channel.values,
            'ip': batch_x.ip.values,
            'hour': batch_x.hour.values,
            xxx
        }
        yield X
    
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
        
    in_hr = Input(shape=[1], name = 'hour')
    emb_hr = Embedding(max_h, 1)(in_hr)
    emb_hr = Reshape((1,))(emb_hr)
   
    fe = concatenate([(emb_app), (emb_ch), (emb_ip), (emb_hr), 
                     (emb_1), (emb_2), (emb_3), (emb_4), (emb_5),
                     (emb_6), (emb_7), (emb_10), 
                     (emb_11), (emb_12), (emb_13), (emb_14), (emb_15),
                     (emb_16), (emb_17), (emb_18), (emb_19), (emb_20),
                     (emb_21), (emb_22), (emb_23), (emb_24), (emb_25),(emb_26), (emb_27) 
                     ,(x)
                     ])

    s_dout = Dropout(0.2)(fe)

    x1 = Dense(500, activation='linear')(s_dout) # 1500 original
    x2 = Dense(500, activation='linear', name="feature")(x1) # 1500 original
    x3 = Dense(500, activation='linear')(x2) # 1500 original
    outputs = Dense(200, activation='linear', name='auto_y')(x3)

    x = concatenate([x1, x2, x3])
    x = Dropout(0.5)(x)
    x = Dense(1500, activation='relu', kernel_regularizer=l2_loss)(x)    

    fe = concatenate([(emb_app), (emb_ch), (emb_ip), (emb_hr), 
                     (emb_1), (emb_2), (emb_3), (emb_4), (emb_5),
                     (emb_6), (emb_7), (emb_10), 
                     (emb_11), (emb_12), (emb_13), (emb_14), (emb_15),
                     (emb_16), (emb_17), (emb_18), (emb_19), (emb_20),
                     (emb_21), (emb_22), (emb_23), (emb_24), (emb_25),(emb_26), (emb_27) 
                     ,(x)
                     ])

    s_dout = Dropout(0.2)(fe)
    z = Dense(2000)(s_dout)
    z = BatchNormalization()(z)    
    z = Activation('relu')(z)    
    z = Dropout(0.5)(z)    
    z = Dense(1000)(z)
    z = BatchNormalization()(z)    
    z = Activation('relu')(z)    
    z = Dropout(0.5)(z)    
    z = Dense(500)(s_dout)
    z = BatchNormalization()(z)    
    z = Activation('relu')(z)    
    z = Dropout(0.5)(z)    

    outp = Dense(1,activation='sigmoid', name = 'y')(z)

    model = Model(inputs=[in_app,in_ch,in_ip,in_hr,
                           in_1, in_2, in_3, in_4, in_5,
                           in_6, in_7, in_10,
                           in_11, in_12, in_13, in_14, in_15,
                           in_16, in_17, in_18, in_19,in_20 ,
                           in_21, in_22, in_23, in_24, in_25,
                           in_26, in_27 
                           ,daeinputs
                           ], outputs=[outp, outputs]
                           )
    return model
    
num_epoch = 10
batch_size = 1024

oobtest = np.zeros((click_id.shape[0],1))
oobval = np.zeros((train.shape[0],1))
valerr = []
cnt = 0
#val_scores = []
cv_r2 = []
nfold = 5
nbag = 1
cols2scale = list(set(test.columns) - set(['ip','app','channel','hour']))
class_weight = {0:.01,1:.99} # magic    
sd = []  ##imp for call back
class LossHistory(Callback):
    def on_train_begin(self, logs={}):
        self.losses = [1,1]
    def on_epoch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
        sd.append(step_decay(len(self.losses)))
        print('lr:', step_decay(len(self.losses)))

def step_decay(epoch):
    if (epoch != 0) & (epoch % 2 ==0):
        lrate = lrate = K.get_value(model.optimizer.lr)
        lrate = lrate * 0.9#**(epoch / 2)#/ (1 + decay_rate * epoch)
        return lrate
    else:
        return K.get_value(model.optimizer.lr)

history=LossHistory()
l_rate=LearningRateScheduler(step_decay)



for i in np.arange(nbag): 
    for seed in [2018]:
        kf = model_selection.StratifiedKFold(n_splits= nfold, shuffle=True, random_state=seed*nbag)
        for dev_index, val_index in kf.split(oobval, y):
            dev_X, val_X = train.iloc[dev_index,:], train.iloc[val_index,:]
#            val_X = val_X.iloc[:1000000]
            dev_y, val_y = y[dev_index], y[val_index]
#            val_y = val_y[:1000000]
#            val_y2 = val_y[:1000000]
            print(dev_X.shape)  
            model = nn()
            earlystopper = EarlyStopping(monitor='val_y_loss', patience=2, verbose=1 , mode='auto')
            plateau = ReduceLROnPlateau(monitor='val_y_loss', factor=0.1, patience=2, verbose=0, mode='auto', epsilon=0.0001, cooldown=0, min_lr=0)
            save_model_path = "nn_sav_"+str(i) +".h5"
            checkpointer = ModelCheckpoint(save_model_path, verbose=1, save_best_only=True)
            optimizer_adam = Adam(lr=0.00001)
            model.compile(loss=['binary_crossentropy','mse'],optimizer=optimizer_adam,loss_weights=[1., 0.1])#,metrics=['accuracy']
            #model.compile(loss='binary_crossentropy',optimizer=optimizer_adam,metrics=['accuracy'])
            gc.collect()
            print("Start Training")
            gen_train = get_keras_data(dev_X, dev_y, batch_size)
            gen_val = get_keras_data(val_X, val_y, batch_size)
            gen_val_pred = get_keras_data_T(val_X, batch_size)
            gen_test_pred = get_keras_data_T(test, batch_size)

            model.fit_generator(generator=gen_train,
                                steps_per_epoch=int(np.ceil(dev_X.shape[0] / batch_size)),
                                class_weight=class_weight,
                                epochs=num_epoch,
                                verbose=1,
                                callbacks=[earlystopper, checkpointer, plateau, history, l_rate],
                                validation_data=gen_val,
                                validation_steps=int(np.ceil(val_X.shape[0] / batch_size)),
                                )
            model = load_model("nn_sav_"+str(i) +".h5")            
            pred = model.predict_generator(generator=gen_val_pred,
                                        steps=np.ceil(val_X.shape[0] / batch_size))[0][0]
            oobval[val_index,:] += pred                                        
            cv_r2.append(roc_auc_score(val_y, pred))    
            print(cv_r2, np.mean(cv_r2),"---", np.std(cv_r2))
            
            oobtest += model.predict_generator(generator=gen_test_pred,
                                        steps=np.ceil(X_test.shape[0] / batch_size))[0][0]

           
            del(dev_X, dev_y, val_X, val_y, model)
            gc.collect()

pred = oobtest / (nfold * nbag)
oobpred = oobval / nbag
joblib.dump(oobpred,'../input/6/train_keras_embedanc_red_ds_5f.pkl')
joblib.dump(pred,'../input/6/test_keras_embedanc_red_ds_5f.pkl')


# Submission
# need to comment next 2 lines
#pred = oobtest
#oobpred = oobval

sub = pd.DataFrame()

test_click_id = click_id
sub['click_id'] = test_click_id.astype('int')
sub['is_attributed'] = pred

sub.to_csv('../output/4/sub_embedanc_keras.csv.gz',compression='gzip', index=False , float_format='%.5g')

                                  
