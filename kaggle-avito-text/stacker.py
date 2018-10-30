import pandas as pd
import numpy as np
from sklearn import model_selection, preprocessing, ensemble
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.externals import joblib

train1 = joblib.load('../l1-train_lgb_bstFtr(tfidf2)_7f.pkl')
test1 = joblib.load('../l1-test_lgb_bstFtr(tfidf2)_7f.pkl')

train2 = joblib.load('../train_keras_all.pkl')
test2 = joblib.load('../test_keras_all.pkl')

train3 = joblib.load('../l1-train_lgb_bstFtr(tfidf2+lblenc)_7f.pkl')
test3 = joblib.load('../l1-test_lgb_bstFtr(tfidf2+lblenc)_7f.pkl')

train4 = joblib.load('../l1-train_weakregr.pkl')
test4 = joblib.load('../l1-test_weakregr.pkl')

y = pd.read_feather('../train_basic_features_woCats.pkl')
y = y.deal_probability.values
test_id = pd.read_feather('../test__basic_features_woCats.pkl')
test_id = test_id.item_id.values


#################################################
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers.advanced_activations import PReLU
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers.recurrent import LSTM
from keras.layers.normalization import BatchNormalization
from keras.layers.core import Dense, Dropout, Activation, Reshape
from keras.layers.embeddings import Embedding
from keras.optimizers import Adam
from sklearn.metrics import mean_squared_error
from keras import backend as K
def root_mean_squared_error(y_true, y_pred):
        return K.sqrt(K.mean(K.square(y_pred - y_true))) 

def nn_model4():
    model = Sequential()
    model.add(Dense(20, input_dim = train.shape[1], init = 'uniform'))#500
    model.add(PReLU())
    model.add(BatchNormalization())
    model.add(Dropout(0.1))#.2
    model.add(Dense(10, init = 'uniform'))#400
    model.add(PReLU())
    model.add(BatchNormalization())
    model.add(Dropout(0.1))#.2
    model.add(Dense(1, init='zero'))
    model.compile(loss = ["MSE"], metrics=[root_mean_squared_error], optimizer = Adam(lr=0.00003, epsilon=1e-08, decay=0.))
    return(model)

train = np.hstack((train1, train2, train3, train4))
test = np.hstack((test1, test2, test3, test4))


oob_tstpred = np.zeros((test.shape[0],1))
oobval = np.zeros((train.shape[0],1))
cv_scores = []
nbag, nfold = 2, 7

seed = 201806 

for x in np.arange(nbag):
    kf = model_selection.KFold(n_splits=nfold, shuffle=True, random_state=201806 * x)    
    for dev_index, val_index in kf.split(train):
        dev_X, val_X = train[dev_index,:], train[val_index,:]
        dev_y, val_y = y[dev_index], y[val_index]
        
        model = nn_model4()
        earlyStopping=EarlyStopping(monitor='val_loss', patience=5, verbose=1, mode='auto')
        checkpointer = ModelCheckpoint(filepath="./weights2XXLK.hdf5", verbose=1, save_best_only=True)
        fit = model.fit(dev_X, dev_y, 
                                  nb_epoch = 100, batch_size=2048,
                                  validation_data=(val_X, val_y),
                                  verbose = 1,callbacks=[earlyStopping,checkpointer]
                                  )
        print("loading weights")
        model =load_model("./weights2XXLK.hdf5",custom_objects={'root_mean_squared_error': root_mean_squared_error})
        print("predicting..")

        preds = model.predict(val_X)#[:,0]
        oobval[val_index] += preds.reshape(-1,1)
        cv_scores.append(mean_squared_error(val_y, preds)**0.5)
        print(cv_scores)
        print(np.mean(cv_scores))
        print(np.std(cv_scores))

        predtst = (model.predict(test))#[:,0]
        oob_tstpred += predtst

tstpred = oob_tstpred/(nfold*nbag)
tstpred[tstpred>1] = 1
tstpred[tstpred<0] = 0

# Making a submission file #
sub_df = pd.DataFrame({"item_id":test_id})
sub_df["deal_probability"] = tstpred
sub_df.to_csv("../output/l2.csv", index=False)    

