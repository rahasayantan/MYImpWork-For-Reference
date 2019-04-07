# define custom R2 metrics for Keras backend
from keras import backend as K

def r2_keras(y_true, y_pred):
    SS_res =  K.sum(K.square( y_true - y_pred )) 
    SS_tot = K.sum(K.square( y_true - K.mean(y_true) ) ) 
    return ( 1 - SS_res/(SS_tot + K.epsilon()) )
    
# base model architecture definition
def model():
    model = Sequential()
    #input layer
    model.add(Dense(input_dims, input_dim=input_dims))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.3))
    # hidden layers
    model.add(Dense(input_dims))
    model.add(BatchNormalization())
    model.add(Activation(act_func))
    model.add(Dropout(0.3))
    
    model.add(Dense(input_dims//2))
    model.add(BatchNormalization())
    model.add(Activation(act_func))
    model.add(Dropout(0.3))
    
    model.add(Dense(input_dims//4, activation=act_func))
    
    # output layer (y_pred)
    model.add(Dense(1, activation='linear'))
    
    # compile this model
    model.compile(loss='mean_squared_error', # one may use 'mean_absolute_error' as alternative
                  optimizer='adam',
                  metrics=[r2_keras] # you can add several if needed
                 )
    
    # Visualize NN architecture
    print(model.summary())
    return model
    
################K2
import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LassoCV
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import RobustScaler
from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense, InputLayer, GaussianNoise
from keras.wrappers.scikit_learn import KerasRegressor

train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')


#
# Data preparation
#
y_train = train['y'].values
id_test = test['ID']

num_train = len(train)
df_all = pd.concat([train, test])
df_all.drop(['ID', 'y'], axis=1, inplace=True)

# One-hot encoding of categorical/strings
df_all = pd.get_dummies(df_all, drop_first=True)

# Sscaling features
scaler = RobustScaler()
df_all = scaler.fit_transform(df_all)

train = df_all[:num_train]
test = df_all[num_train:]

# Keep only the most contributing features
sfm = SelectFromModel(LassoCV())
sfm.fit(train, y_train)
train = sfm.transform(train)
test = sfm.transform(test)

print ('Number of features : %d' % train.shape[1])

def r2_keras(y_true, y_pred):
    SS_res =  K.sum(K.square( y_true - y_pred ))
    SS_tot = K.sum(K.square( y_true - K.mean(y_true) ) )
    return ( 1 - SS_res/(SS_tot + K.epsilon()) )

def build_model_fn(neurons=20, noise=0.25):
    model = Sequential()
    model.add(InputLayer(input_shape=(train.shape[1],)))
    model.add(GaussianNoise(noise))
    model.add(Dense(neurons, activation='tanh'))
    model.add(Dense(1, activation='linear'))
    model.compile(loss='mean_squared_error', optimizer='nadam', metrics=[r2_keras])
    return model
    
    
#
# Tuning model parameters
#
model = KerasRegressor(build_fn=build_model_fn, epochs=75, verbose=0)

gsc = GridSearchCV(
    estimator=model,
    param_grid={
        #'neurons': range(18,31,4),
        'noise': [x/20.0 for x in range(3, 7)],
    },
    #scoring='r2',
    scoring='neg_mean_squared_error',
    cv=5
)

grid_result = gsc.fit(train, y_train)

print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
for test_mean, test_stdev, train_mean, train_stdev, param in zip(
        grid_result.cv_results_['mean_test_score'],
        grid_result.cv_results_['std_test_score'],
        grid_result.cv_results_['mean_train_score'],
        grid_result.cv_results_['std_train_score'],
        grid_result.cv_results_['params']):
    print("Train: %f (%f) // Test : %f (%f) with: %r" % (train_mean, train_stdev, test_mean, test_stdev, param))
    
    
#
# Train model with best params for submission
#
model = build_model_fn(**grid_result.best_params_)

model.fit(train, y_train, epochs=75, verbose=2)

y_test = model.predict(test).flatten()

df_sub = pd.DataFrame({'ID': id_test, 'y': y_test})
df_sub.to_csv('mercedes-submission.csv', index=False)

#########################
import pandas as pd
import numpy as np
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.decomposition import PCA, FastICA
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import make_pipeline, Pipeline, _name_estimators
from sklearn.linear_model import ElasticNet, ElasticNetCV
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import r2_score
from sklearn.base import BaseEstimator, TransformerMixin
import xgboost as xgb


train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')

y_train = train['y'].values
y_mean = np.mean(y_train)
id_test = test['ID']

num_train = len(train)
df_all = pd.concat([train, test])
df_all.drop(['ID', 'y'], axis=1, inplace=True)

# One-hot encoding of categorical/strings
df_all = pd.get_dummies(df_all, drop_first=True)

train = df_all[:num_train]
test = df_all[num_train:]


class AddColumns(BaseEstimator, TransformerMixin):
    def __init__(self, transform_=None):
        self.transform_ = transform_

    def fit(self, X, y=None):
        self.transform_.fit(X, y)
        return self

    def transform(self, X, y=None):
        xform_data = self.transform_.transform(X, y)
        return np.append(X, xform_data, axis=1)


class LogExpPipeline(Pipeline):
    def fit(self, X, y):
        super(LogExpPipeline, self).fit(X, np.log1p(y))

    def predict(self, X):
        return np.expm1(super(LogExpPipeline, self).predict(X))

#
# Model/pipeline with scaling,pca,svm
#
svm_pipe = LogExpPipeline(_name_estimators([RobustScaler(),
                                            PCA(),
                                            SVR(kernel='rbf', C=1.0, epsilon=0.05)]))
                                            
# results = cross_val_score(svm_pipe, train, y_train, cv=5, scoring='r2')
# print("SVM score: %.4f (%.4f)" % (results.mean(), results.std()))
# exit()
                                            
#
# Model/pipeline with scaling,pca,ElasticNet
#
en_pipe = LogExpPipeline(_name_estimators([RobustScaler(),
                                           PCA(n_components=125),
                                           ElasticNet(alpha=0.001, l1_ratio=0.1)]))

#
# XGBoost model
#
xgb_model = xgb.sklearn.XGBRegressor(max_depth=4, learning_rate=0.005, subsample=0.921,
                                     objective='reg:linear', n_estimators=1300, base_score=y_mean)
                                     
xgb_pipe = Pipeline(_name_estimators([AddColumns(transform_=PCA(n_components=10)),
                                      AddColumns(transform_=FastICA(n_components=10, max_iter=500)),
                                      xgb_model]))

# results = cross_val_score(xgb_model, train, y_train, cv=5, scoring='r2')
# print("XGB score: %.4f (%.4f)" % (results.mean(), results.std()))


#
# Random Forest
#
rf_model = RandomForestRegressor(n_estimators=250, n_jobs=4, min_samples_split=25,
                                 min_samples_leaf=25, max_depth=3)

# results = cross_val_score(rf_model, train, y_train, cv=5, scoring='r2')
# print("RF score: %.4f (%.4f)" % (results.mean(), results.std()))


#
# Now the training and stacking part.  In previous version i just tried to train each model and
# find the best combination, that lead to a horrible score (Overfit?).  Code below does out-of-fold
# training/predictions and then we combine the final results.
#
# Read here for more explanation (This code was borrowed/adapted) :
#

class Ensemble(object):
    def __init__(self, n_splits, stacker, base_models):
        self.n_splits = n_splits
        self.stacker = stacker
        self.base_models = base_models

    def fit_predict(self, X, y, T):
        X = np.array(X)
        y = np.array(y)
        T = np.array(T)

        folds = list(KFold(n_splits=self.n_splits, shuffle=True, random_state=2016).split(X, y))

        S_train = np.zeros((X.shape[0], len(self.base_models)))
        S_test = np.zeros((T.shape[0], len(self.base_models)))
        for i, clf in enumerate(self.base_models):

            S_test_i = np.zeros((T.shape[0], self.n_splits))

            for j, (train_idx, test_idx) in enumerate(folds):
                X_train = X[train_idx]
                y_train = y[train_idx]
                X_holdout = X[test_idx]
                y_holdout = y[test_idx]

                clf.fit(X_train, y_train)
                y_pred = clf.predict(X_holdout)[:]

                print ("Model %d fold %d score %f" % (i, j, r2_score(y_holdout, y_pred)))

                S_train[test_idx, i] = y_pred
                S_test_i[:, j] = clf.predict(T)[:]
            S_test[:, i] = S_test_i.mean(axis=1)

        # results = cross_val_score(self.stacker, S_train, y, cv=5, scoring='r2')
        # print("Stacker score: %.4f (%.4f)" % (results.mean(), results.std()))
        # exit()

        self.stacker.fit(S_train, y)
        res = self.stacker.predict(S_test)[:]
        return res

stack = Ensemble(n_splits=5,
                 #stacker=ElasticNetCV(l1_ratio=[x/10.0 for x in range(1,10)]),
                 stacker=ElasticNet(l1_ratio=0.1, alpha=1.4),
                 base_models=(svm_pipe, en_pipe, xgb_pipe, rf_model))

y_test = stack.fit_predict(train, y_train, test)

df_sub = pd.DataFrame({'ID': id_test, 'y': y_test})
df_sub.to_csv('submission.csv', index=False)

#############################
'''This example demonstrates the use of Convolution1D for text classification.
Gets to 0.89 test accuracy after 2 epochs.
90s/epoch on Intel i5 2.4Ghz CPU.
10s/epoch on Tesla K40 GPU.
'''

from __future__ import print_function

from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding
from keras.layers import Conv1D, GlobalMaxPooling1D
from keras.datasets import imdb

# set parameters:
max_features = 5000
maxlen = 400
batch_size = 32
embedding_dims = 50
filters = 250
kernel_size = 3
hidden_dims = 250
epochs = 2

print('Loading data...')
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)
print(len(x_train), 'train sequences')
print(len(x_test), 'test sequences')

print('Pad sequences (samples x time)')
x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = sequence.pad_sequences(x_test, maxlen=maxlen)
print('x_train shape:', x_train.shape)
print('x_test shape:', x_test.shape)

print('Build model...')
model = Sequential()

# we start off with an efficient embedding layer which maps
# our vocab indices into embedding_dims dimensions
model.add(Embedding(max_features,
                    embedding_dims,
                    input_length=maxlen))
model.add(Dropout(0.2))

# we add a Convolution1D, which will learn filters
# word group filters of size filter_length:
model.add(Conv1D(filters,
                 kernel_size,
                 padding='valid',
                 activation='relu',
                 strides=1))
# we use max pooling:
model.add(GlobalMaxPooling1D())

# We add a vanilla hidden layer:
model.add(Dense(hidden_dims))
model.add(Dropout(0.2))
model.add(Activation('relu'))

# We project onto a single unit output layer, and squash it with a sigmoid:
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          validation_data=(x_test, y_test))        
