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
class GaussRankScaler():

	def __init__( self ):
		self.epsilon = 0.001
		self.lower = -1 + self.epsilon
		self.upper =  1 - self.epsilon
		self.range = self.upper - self.lower

	def fit_transform( self, X ):
	
		i = np.argsort( X, axis = 0 )
		j = np.argsort( i, axis = 0 )

		assert ( j.min() == 0 ).all()
		assert ( j.max() == len( j ) - 1 ).all()
		
		j_range = len( j ) - 1
		self.divider = j_range / self.range
		
		transformed = j / self.divider
		transformed = transformed - self.upper
		transformed = erfinv( transformed )
		
		return transformed

train = joblib.load('../input/6/train_nn_ts_2d_smallHr_ds.pkl')
test = joblib.load('../input/6/test_nn_ts_2d_smallHr_ds.pkl')
max_app = np.max([train['app'].max(), test['app'].max()])+1
max_ch = np.max([train['channel'].max(), test['channel'].max()])+1
max_ip = np.max([train['ip'].max(), test['ip'].max()])+1
max_h = np.max([train['hour'].max(), test['hour'].max()])+1
y = joblib.load('../input/6/y_countenc_ts_2d_smallHr_ds.pkl')
click_id = joblib.load('../input/6/click_id_countenc_ts_2d_smallHr_ds.pkl')


'''train = joblib.load('../input/6/train_countenc_ts_2d_smallHr_ds2.pkl')
test = joblib.load('../input/6/test_countenc_ts_2d_smallHr_ds2.pkl')
y = joblib.load('../input/6/y_countenc_ts_2d_smallHr_ds.pkl')
click_id = joblib.load('../input/6/click_id_countenc_ts_2d_smallHr_ds.pkl')

gc.collect()
assert(set(train.columns) - set(test.columns) == set())
print("Read Complete..")

cols2scale = list(set(test.columns) - set(['ip','app','channel','hour']))

le = GaussRankScaler()#preprocessing.MinMaxScaler() # normalization method of original is rank gaussian
for vcol in cols2scale:
    x = le.fit_transform(np.vstack((train[vcol].reshape(-1,1), test[vcol].reshape(-1,1))))
    train[vcol] = x[:train.shape[0]]
    test[vcol] = x[train.shape[0]:]
    if train[vcol].dtype == 'float64':
        train[vcol] = train[vcol].astype('float32')
        test[vcol] = test[vcol].astype('float32')


cols2enc = ['app', 'channel', 'ip', 'hour']
le = preprocessing.LabelEncoder() 
for vcol in cols2enc:
    le.fit(np.vstack((train[vcol].reshape(-1,1), test[vcol].reshape(-1,1))))
    train[vcol] = le.transform(train[vcol].reshape(-1,1))
    test[vcol] = le.transform(test[vcol].reshape(-1,1))

max_app = np.max([train['app'].max(), test['app'].max()])+1
max_ch = np.max([train['channel'].max(), test['channel'].max()])+1
max_ip = np.max([train['ip'].max(), test['ip'].max()])+1
max_h = np.max([train['hour'].max(), test['hour'].max()])+1



le = preprocessing.MinMaxScaler() 
for vcol in cols2scale:
    le.fit(np.vstack((train[vcol].reshape(-1,1), test[vcol].reshape(-1,1))))
    train[vcol] = le.transform(train[vcol].reshape(-1,1))
    test[vcol] = le.transform(test[vcol].reshape(-1,1))
    if train[vcol].dtype == 'float64':
        train[vcol] = train[vcol].astype('float32')
        test[vcol] = test[vcol].astype('float32')

'''

print ('neural network....')
#import plaidml.keras
#plaidml.keras.install_backend()
from keras.layers import Input, Embedding, Dense, Flatten, Dropout, concatenate, Reshape, Concatenate
from keras.layers import BatchNormalization, SpatialDropout1D, Conv1D
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
        #batch_x = dataset1.iloc[index_array[current_index: current_index + current_batch_size],:]
        #batch_y = y[index_array[current_index: current_index + current_batch_size]]
        batch_x = dataset1.iloc[current_index: current_index + current_batch_size,:]

        batch_y = y[current_index: current_index + current_batch_size]
        
        new_batch = batch_x[cols2scale].values        
        index_array = np.random.permutation(new_batch.shape[0])
        batch2 = new_batch[index_array,:]
        #print(batch2.shape)     
        for i in range(new_batch.shape[0]):# ideally should be done for every row, but thats too slow
            swap_idx = np.random.choice(25, 4, replace=False)# columns swapping to gen noise. should br .1/.15 * num column 
            new_batch[i, swap_idx] = batch2[i, swap_idx]
        # due to slowness instead of doing at row level, i am doing at columns level
        #swap_idx = np.random.choice(num_value, num_swap, replace=False)
        #new_batch[:, swap_idx] = batch2[:, swap_idx]
    
        X = {
            'app': batch_x.app.values,
            'ch': batch_x.channel.values,
            'ip': batch_x.ip.values,
            'hour': batch_x.hour.values,
            'time_since_last_click_3': batch_x.time_since_last_click_3.values,
            'cumsum2': batch_x.cumsum2.values,
            'cumsum4': batch_x.cumsum4.values,
            'cumsum2_hr': batch_x.cumsum2_hr.values,
            'cumsum3_hr': batch_x.cumsum3_hr.values,
            'cumsum4_hr': batch_x.cumsum4_hr.values,
            'cumsum1_hr': batch_x.cumsum1_hr.values,
            'y_mean41': batch_x.y_mean41.values,
            'y_mean42': batch_x.y_mean42.values,
            'y_mean43': batch_x.y_mean43.values,
            'y_mean44': batch_x.y_mean44.values,
            'y_mean45': batch_x.y_mean45.values,
            'y_mean432': batch_x.y_mean432.values,
            'y_lenc41': batch_x.y_lenc41.values,
            'y_lenc42': batch_x.y_lenc42.values,
            'y_lenc43': batch_x.y_lenc43.values,
            'y_lenc45': batch_x.y_lenc45.values,
            'y_lenc431': batch_x.y_lenc431.values,
            'y_lenc432': batch_x.y_lenc432.values,
            'y_lenc21': batch_x.y_lenc21.values,
            'y_lenc22': batch_x.y_lenc22.values,
            'y_lenc23': batch_x.y_lenc23.values,   
            'y_lenc25': batch_x.y_lenc25.values,
            'y_lenc231': batch_x.y_lenc231.values,
            'y_lenc232': batch_x.y_lenc232.values
            ,'autoenc': new_batch
        }
        
        Y = { 
            'y' : batch_y
            ,'auto_y' : batch_x[cols2scale].values
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
        #batch_x = dataset1.iloc[index_array[current_index: current_index + current_batch_size],:]
        #batch_y = y[index_array[current_index: current_index + current_batch_size]]
        batch_x = dataset1.iloc[current_index: current_index + current_batch_size,:]
        batch_y = y[current_index: current_index + current_batch_size]
        
        new_batch = batch_x[cols2scale].values        
        index_array = np.random.permutation(new_batch.shape[0])
        batch2 = new_batch[index_array,:]
        
        for i in range(new_batch.shape[0]):# ideally should be done for every row, but thats too slow
            swap_idx = np.random.choice(25, 4, replace=False)# $ columns i will be swapping to gen noise. should br .1/.15 * num column 
            new_batch[i, swap_idx] = batch2[i, swap_idx]
        # due to slowness instead of doing at row level, i am doing at columns level
        #swap_idx = np.random.choice(num_value, num_swap, replace=False)
        #new_batch[:, swap_idx] = batch2[:, swap_idx]
    
        X = {
            'app': batch_x.app.values,
            'ch': batch_x.channel.values,
            'ip': batch_x.ip.values,
            'hour': batch_x.hour.values,
            'time_since_last_click_3': batch_x.time_since_last_click_3.values,
            'cumsum2': batch_x.cumsum2.values,
            'cumsum4': batch_x.cumsum4.values,
            'cumsum2_hr': batch_x.cumsum2_hr.values,
            'cumsum3_hr': batch_x.cumsum3_hr.values,
            'cumsum4_hr': batch_x.cumsum4_hr.values,
            'cumsum1_hr': batch_x.cumsum1_hr.values,
            'y_mean41': batch_x.y_mean41.values,
            'y_mean42': batch_x.y_mean42.values,
            'y_mean43': batch_x.y_mean43.values,
            'y_mean44': batch_x.y_mean44.values,
            'y_mean45': batch_x.y_mean45.values,
            'y_mean432': batch_x.y_mean432.values,
            'y_lenc41': batch_x.y_lenc41.values,
            'y_lenc42': batch_x.y_lenc42.values,
            'y_lenc43': batch_x.y_lenc43.values,
            'y_lenc45': batch_x.y_lenc45.values,
            'y_lenc431': batch_x.y_lenc431.values,
            'y_lenc432': batch_x.y_lenc432.values,
            'y_lenc21': batch_x.y_lenc21.values,
            'y_lenc22': batch_x.y_lenc22.values,
            'y_lenc23': batch_x.y_lenc23.values,   
            'y_lenc25': batch_x.y_lenc25.values,
            'y_lenc231': batch_x.y_lenc231.values,
            'y_lenc232': batch_x.y_lenc232.values
            ,'autoenc': new_batch
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
   
    in_1 = Input(shape=[1], name = 'time_since_last_click_3')
    emb_1 = Dense(1)(in_1) 
    in_2 = Input(shape=[1], name = 'cumsum2')
    emb_2 = Dense(1)(in_2) 
    in_3 = Input(shape=[1], name = 'cumsum4')
    emb_3 = Dense(1)(in_3) 
    in_4 = Input(shape=[1], name = 'cumsum2_hr')
    emb_4 = Dense(1)(in_4) 
    in_5 = Input(shape=[1], name = 'cumsum3_hr')
    emb_5 = Dense(1)(in_5) 
    in_6 = Input(shape=[1], name = 'cumsum4_hr')
    emb_6 = Dense(1)(in_6)  
    in_7 = Input(shape=[1], name = 'cumsum1_hr')
    emb_7 = Dense(1)(in_7)  
    in_10 = Input(shape=[1], name = 'y_mean41')
    emb_10 = Dense(1)(in_10) 
    in_11 = Input(shape=[1], name = 'y_mean42')
    emb_11 = Dense(1)(in_11)  
    in_12 = Input(shape=[1], name = 'y_mean43')
    emb_12 = Dense(1)(in_12)             
    in_13 = Input(shape=[1], name = 'y_mean44')
    emb_13 = Dense(1)(in_13) 
    in_14 = Input(shape=[1], name = 'y_mean45')
    emb_14 = Dense(1)(in_14)  
    in_15 = Input(shape=[1], name = 'y_mean432')
    emb_15 = Dense(1)(in_15) 
    in_16 = Input(shape=[1], name = 'y_lenc41')
    emb_16 = Dense(1)(in_16) 
    in_17 = Input(shape=[1], name = 'y_lenc42')
    emb_17 = Dense(1)(in_17)  
    in_18 = Input(shape=[1], name = 'y_lenc43')
    emb_18 = Dense(1)(in_18) 
    in_19 = Input(shape=[1], name = 'y_lenc45')
    emb_19 = Dense(1)(in_19)  
    in_20 = Input(shape=[1], name = 'y_lenc431')
    emb_20 = Dense(1)(in_20)  
    in_21 = Input(shape=[1], name = 'y_lenc432')
    emb_21 = Dense(1)(in_21)  
    in_22 = Input(shape=[1], name = 'y_lenc21')
    emb_22 = Dense(1)(in_22)  
    in_23 = Input(shape=[1], name = 'y_lenc22')
    emb_23 = Dense(1)(in_23)  
    in_24 = Input(shape=[1], name = 'y_lenc23')
    emb_24 = Dense(1)(in_24)  
    in_25 = Input(shape=[1], name = 'y_lenc25')
    emb_25 = Dense(1)(in_25)  
    in_26 = Input(shape=[1], name = 'y_lenc231')
    emb_26 = Dense(1)(in_26)  
    in_27 = Input(shape=[1], name = 'y_lenc232')
    emb_27 = Dense(1)(in_27)  

    daeinputs = Input(shape=[25], name = 'autoenc')
    x1 = Dense(50, activation='linear')(daeinputs) # 1500 original
    x2 = Dense(50, activation='linear', name="feature")(x1) # 1500 original
    x3 = Dense(50, activation='linear')(x2) # 1500 original
    outputs = Dense(25, activation='linear', name='auto_y')(x3)
#    auto = Model(inputs=daeinputs, outputs=outputs)
#    auto.compile(optimizer='adam', loss='mse')

    l2_loss = l2(0.05)

    x = concatenate([x1, x2, x3])
    x = Dropout(0.5)(x)
    x = Dense(150, activation='relu', kernel_regularizer=l2_loss)(x)    

    fe = concatenate([(emb_app), (emb_ch), (emb_ip), (emb_hr), 
                     (emb_1), (emb_2), (emb_3), (emb_4), (emb_5),
                     (emb_6), (emb_7), (emb_10), 
                     (emb_11), (emb_12), (emb_13), (emb_14), (emb_15),
                     (emb_16), (emb_17), (emb_18), (emb_19), (emb_20),
                     (emb_21), (emb_22), (emb_23), (emb_24), (emb_25),(emb_26), (emb_27) 
                     ,(x)
                     ])

    s_dout = Dropout(0.2)(fe)#SpatialDropout1D(0.2)(fe)

    z = Dropout(0.5)(Dense(300,activation='relu')(s_dout))#(s_dout))#
    z = Dropout(0.5)(Dense(300,activation='relu')(z))
    z = Dropout(0.5)(Dense(150,activation='relu')(z))

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

                                  
