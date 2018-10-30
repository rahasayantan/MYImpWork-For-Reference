import os; os.environ['OMP_NUM_THREADS'] = '2'
from sklearn.ensemble import ExtraTreesRegressor
import nltk
nltk.data.path.append("/media/sayantan/Personal/nltk_data")
from nltk.stem.snowball import RussianStemmer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer as Tfidf
from sklearn.metrics import mean_squared_log_error
from sklearn.model_selection import KFold
from sklearn.externals import joblib
from scipy.sparse import hstack, csr_matrix
import lightgbm as lgb
from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np
import tensorflow as tf
import gc
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder, MinMaxScaler
from sklearn import model_selection
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import Ridge, Lasso, HuberRegressor, ElasticNet, BayesianRidge, LinearRegression

train_x1 = pd.read_feather('../train_imagetop_targetenc.pkl')
test_x1 = pd.read_feather('../test_imagetop_targetenc.pkl')
train_x1.fillna(-1, inplace = True)
test_x1.fillna(-1, inplace = True)
for col in train_x1.columns:
    lbl = MinMaxScaler()
    X = np.hstack((train_x1[col].fillna(-1).values, test_x1[col].fillna(-1).values)).reshape(-1,1)
    lbl.fit(X)
    train_x1[col] = lbl.transform(train_x1[col].fillna(-1).values.reshape(-1,1))
    test_x1[col] = lbl.transform(test_x1[col].fillna(-1).values.reshape(-1,1))

train_x2 = pd.read_feather('../train_itemseq_targetenc.pkl')
test_x2 = pd.read_feather('../test_itemseq_targetenc.pkl')
train_x2.fillna(-1, inplace = True)
test_x2.fillna(-1, inplace = True)
for col in train_x2.columns:
    lbl = MinMaxScaler()
    X = np.hstack((train_x2[col].fillna(-1).values, test_x2[col].fillna(-1).values)).reshape(-1,1)
    lbl.fit(X)
    train_x2[col] = lbl.transform(train_x2[col].fillna(-1).values.reshape(-1,1))
    test_x2[col] = lbl.transform(test_x2[col].fillna(-1).values.reshape(-1,1))


region = joblib.load("../region_onehot.pkl")
city = joblib.load("../city_onehot.pkl")
parent_category_name = joblib.load("../parent_category_name_onehot.pkl")
category_name = joblib.load("../category_name_onehot.pkl")
user_type = joblib.load("../user_type_onehot.pkl")

train_df = pd.read_feather('../train_basic_features_woCats.pkl')
test_df = pd.read_feather('../test__basic_features_woCats.pkl')
y = train_df.deal_probability.values
test_id = test_df.item_id.values
train_df.drop(['deal_probability'], axis = 'columns', inplace = True)
test_df.drop(['deal_probability'], axis = 'columns', inplace = True)
item_id = test_df.item_id.values
############################
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
'''
MAX_NUM_OF_WORDS = 100000
TIT_MAX_SEQUENCE_LENGTH = 100

df = pd.concat((train_df, test_df), axis = 'rows')

tokenizer = Tokenizer(num_words=MAX_NUM_OF_WORDS)
tokenizer.fit_on_texts(df['title'].tolist())
sequences = tokenizer.texts_to_sequences(df['title'].tolist())
titleSequences = pad_sequences(sequences, maxlen=TIT_MAX_SEQUENCE_LENGTH)
joblib.dump(titleSequences, "../titleSequences.pkl")

MAX_NUM_OF_WORDS = 10000
TIT_MAX_SEQUENCE_LENGTH = 20

tokenizer = Tokenizer(num_words=MAX_NUM_OF_WORDS)
tokenizer.fit_on_texts(df['params'].tolist())
sequences = tokenizer.texts_to_sequences(df['params'].tolist())
titleSequences = pad_sequences(sequences, maxlen=TIT_MAX_SEQUENCE_LENGTH)
joblib.dump(titleSequences, "../paramSequences.pkl")
MAX_NUM_OF_WORDS = 100000
TIT_MAX_SEQUENCE_LENGTH = 100

tokenizer = Tokenizer(num_words=MAX_NUM_OF_WORDS)
tokenizer.fit_on_texts(df['description'].tolist())
sequences = tokenizer.texts_to_sequences(df['description'].tolist())
titleSequences = pad_sequences(sequences, maxlen=TIT_MAX_SEQUENCE_LENGTH)
joblib.dump(titleSequences, "../descSequences.pkl")
'''
titleSequences = joblib.load("../titleSequences.pkl")
paramSequences = joblib.load("../paramSequences.pkl")
descSequences = joblib.load("../descSequences.pkl")

ts_tr, ts_te = titleSequences[:train_df.shape[0]], titleSequences[train_df.shape[0]:]
pa_tr, pa_te = paramSequences[:train_df.shape[0]], paramSequences[train_df.shape[0]:]
desc_tr, desc_te = descSequences[:train_df.shape[0]], descSequences[train_df.shape[0]:]
del(titleSequences, paramSequences, descSequences); gc.collect()
## get the label encoding cats for encoding
train_df2=pd.read_feather('../train_basic_features_lblencCats.pkl')
test_df2=pd.read_feather('../test__basic_features_lblencCats.pkl')
catCols = ['user_id', 'region', 'city', 'parent_category_name',
       'category_name', 'user_type', 'param_1', 'param_2', 'param_3']
train_df2 = train_df2[catCols]
test_df2 = test_df2[catCols]
gc.collect()
###############################

cols_to_drop =["item_id", "title", "description", "activation_date", "image", "params",
               "title_clean", "desc_clean", "params_clean",
               "get_nouns_title","get_nouns_desc",
               "get_adj_title","get_adj_desc",
               "get_verb_title","get_verb_desc",
               "monthday", "price"
               ]

train_df.drop(cols_to_drop, axis = 'columns', inplace = True)
test_df.drop(cols_to_drop, axis = 'columns', inplace = True)
gc.collect()
train_df.fillna(-1, inplace = True)
test_df.fillna(-1, inplace = True)

for col in train_df.columns:
    lbl = MinMaxScaler()
    X = np.hstack((train_df[col].fillna(-1).values, test_df[col].fillna(-1).values)).reshape(-1,1)
    lbl.fit(X)
    train_df[col] = lbl.transform(train_df[col].fillna(-1).values.reshape(-1,1))
    test_df[col] = lbl.transform(test_df[col].fillna(-1).values.reshape(-1,1))

week = joblib.load("../activation_weekday_onehot.pkl")
train_df.drop(['activation_weekday'], axis = 'columns', inplace = True)
test_df.drop(['activation_weekday'], axis = 'columns', inplace = True)

param_train_tfidf, param_test_tfidf = joblib.load("../params_tfidf.pkl")
title_train_tfidf, title_test_tfidf = joblib.load("../title_tfidf.pkl")
desc_train_tfidf, desc_test_tfidf = joblib.load("../desc_tfidf.pkl")

region_train,region_test = region[:train_df.shape[0]],region[train_df.shape[0]:]
pcn_train, pcn_test = parent_category_name[:train_df.shape[0]], parent_category_name[train_df.shape[0]:] 
cn_train, cn_test = category_name[:train_df.shape[0]],category_name[train_df.shape[0]:],  
ut_train, ut_test = user_type[:train_df.shape[0]], user_type[train_df.shape[0]:]
city_train, city_test = city[:train_df.shape[0]], city[train_df.shape[0]:]
week_train, week_test = week[:train_df.shape[0]],week[train_df.shape[0]:]
del(region, parent_category_name, category_name, user_type, city, week); gc.collect()

param_train_tfidf, param_test_tfidf = param_train_tfidf.tocsr(), param_test_tfidf.tocsr()
title_train_tfidf, title_test_tfidf = title_train_tfidf.tocsr(), title_test_tfidf.tocsr()
desc_train_tfidf, desc_test_tfidf = desc_train_tfidf.tocsr(), desc_test_tfidf.tocsr()

tit_train_char, tit_test_char = joblib.load('../title_chargram_tfidf.pkl')
par_train_char, par_test_char = joblib.load('../param_chargram_tfidf.pkl')
cat_train_char, cat_test_char = joblib.load('../cat_chargram_tfidf.pkl')

train_2=pd.read_feather('../train_cat_targetenc.pkl')
test_2=pd.read_feather('../test_cat_targetenc.pkl')
catCols = ['user_id', 'region', 'city', 'parent_category_name',
       'category_name', 'user_type']
train_2.drop(catCols, axis = 'columns', inplace=True)
test_2.drop(catCols, axis = 'columns', inplace=True)

train_2.fillna(-1, inplace = True)
test_2.fillna(-1, inplace = True)

for col in train_2.columns:
    lbl = MinMaxScaler()
    X = np.hstack((train_2[col].fillna(-1).values, test_2[col].fillna(-1).values)).reshape(-1,1)
    lbl.fit(X)
    train_2[col] = lbl.transform(train_2[col].fillna(-1).values.reshape(-1,1))
    test_2[col] = lbl.transform(test_2[col].fillna(-1).values.reshape(-1,1))

train_3=pd.read_feather('../train_kag_agg_ftr.ftr')
test_3=pd.read_feather('../test_kag_agg_ftr.ftr')

for col in train_3.columns:
    lbl = MinMaxScaler()
    X = np.hstack((train_3[col].fillna(-1).values, test_3[col].fillna(-1).values)).reshape(-1,1)
    lbl.fit(X)
    train_3[col] = lbl.transform(train_3[col].fillna(-1).values.reshape(-1,1))
    test_3[col] = lbl.transform(test_3[col].fillna(-1).values.reshape(-1,1))

train_4 = joblib.load('../l1-train_weakregr.pkl')
test_4 = joblib.load('../l1-test_weakregr.pkl')
            
def get_keras_sparse(df1, df2, df3, df4, df5, df6, df7, df8, df9, df10, df11, df12, df13, df14, df15, df16, df17,
                     df18, df19, df20, df21, df22, df23, df24, df25, df26, df27, df28#, df29
                     ):
    X = {'in1': df1,
        'in2': df2,
        'in3': df3,
        'in4': df4,
        'in5': df5,
        'in6': df6,
        'in7': df7,
        'in8': df8,
        'in9': df9,
        'in10': df10,                                                               
        'in11': df11,
        'in12': df12,
        'in13': df13,
        'in14': df14,
        'in15': df15,
        'in16': df16,
        'in17': df17,
        
        'in18': df18,
        'in19': df19,
        'in20': df20,
                                
        'in21': df21,
        'in22': df22,
        'in23': df23,
        'in24': df24,
        'in25': df25,
        'in26': df26,
        'in27': df27,
        'in28': df28#,
        #'in29': df29
    }
    return X

def get_keras_sparse2(df1, df9, df11, df12, df13, df14, 
                     df18, df19, df20, df21, df22, df23, df24, df25, df26, df27, df28#, df29
                     ):
    X = {'in1': df1,
        'in9': df9,
        'in11': df11,
        'in12': df12,
        'in13': df13,
        'in14': df14,
        
        'in18': df18,
        'in19': df19,
        'in20': df20,
                                
        'in21': df21,
        'in22': df22,
        'in23': df23,
        'in24': df24,
        'in25': df25,
        'in26': df26,
        'in27': df27,
        'in28': df28#,
        #'in29': df29
    }
    return X

import keras as ks
from keras.layers import Input, Embedding, Dense, Flatten, Dropout, concatenate, Reshape, Concatenate, BatchNormalization
from keras.layers import BatchNormalization, SpatialDropout1D, Conv1D, Activation,Bidirectional
from keras.callbacks import Callback
from keras.models import Model
from keras.optimizers import Adam
from keras.models import Model, load_model
from  keras.regularizers import l2
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, LearningRateScheduler, Callback
from sklearn import model_selection
from keras.layers import Input, Embedding, Dense, Flatten, Dropout, concatenate, Reshape, Concatenate, BatchNormalization, GRU
from keras import backend as K
from keras.engine import InputSpec, Layer
from keras import regularizers

class Attention(Layer):

    def __init__(self, CONTEXT_DIM = 50, regularizer=regularizers.l2(1e-10), **kwargs):
        self.regularizer = regularizer
        self.supports_masking = True
        self.CONTEXT_DIM = CONTEXT_DIM
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3        
        self.W = self.add_weight(name='W',
                                 shape=(input_shape[-1], self.CONTEXT_DIM),
                                 initializer='normal',
                                 trainable=True, 
                                 regularizer=self.regularizer)
        self.b = self.add_weight(name='b',
                                 shape=(self.CONTEXT_DIM,),
                                 initializer='normal',
                                 trainable=True, 
                                 regularizer=self.regularizer)
        self.u = self.add_weight(name='u',
                                 shape=(self.CONTEXT_DIM,),
                                 initializer='normal',
                                 trainable=True, 
                                 regularizer=self.regularizer)        
        super(Attention, self).build(input_shape)

    @staticmethod
    def softmax(x, dim):
        """Computes softmax along a specified dim. Keras currently lacks this feature.
        """
        if K.backend() == 'tensorflow':
            import tensorflow as tf
            return tf.nn.softmax(x, dim)
        elif K.backend() == 'theano':
            # Theano cannot softmax along an arbitrary dim.
            # So, we will shuffle `dim` to -1 and un-shuffle after softmax.
            perm = np.arange(K.ndim(x))
            perm[dim], perm[-1] = perm[-1], perm[dim]
            x_perm = K.permute_dimensions(x, perm)
            output = K.softmax(x_perm)

            # Permute back
            perm[dim], perm[-1] = perm[-1], perm[dim]
            output = K.permute_dimensions(x, output)
            return output
        else:
            raise ValueError("Backend '{}' not supported".format(K.backend()))

    def call(self, x, mask=None):
        ut = K.tanh(K.bias_add(K.dot(x, self.W), self.b)) * self.u

        # Collapse `attention_dims` to 1. This indicates the weight for each time_step.
        ut = K.sum(ut, axis=-1, keepdims=True)

        # Convert those weights into a distribution but along time axis.
        # i.e., sum of alphas along `time_steps` axis should be 1.
        self.at = self.softmax(ut, dim=1)
        if mask is not None:
            self.at *= K.cast(K.expand_dims(mask, -1), K.floatx())

        # Weighted sum along `time_steps` axis.
        return K.sum(x * self.at, axis=-2)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1])
    
    def get_config(self):
        config = {}
        base_config = super(Attention, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_mask(self, inputs, mask):
        return None

def nn_model5():
    model_in1 = ks.Input(shape=(train_df.shape[1],),name='in1', dtype='float32')
    out1 = ks.layers.Dense(train_df.shape[1])(model_in1)        
#    out1 = Dropout(0.4)(out1)
    
    model_in9 = ks.Input(shape=(week_train.shape[1],),name='in9', dtype='float32', sparse=True)            
    out9 = ks.layers.Dense(7)(model_in9) 
#    out9 = Dropout(0.4)(out9)
    
    model_in11 = ks.Input(shape=(train_x1.shape[1],),name='in11', dtype='float32')            
    out11 = ks.layers.Dense(train_x1.shape[1])(model_in11) 
#    out11 = Dropout(0.4)(out11)
    
    model_in12 = ks.Input(shape=(train_x2.shape[1],),name='in12', dtype='float32')            
    out12 = ks.layers.Dense(train_x2.shape[1])(model_in12) 
#    out12 = Dropout(0.4)(out12)
    
    model_in13 = ks.Input(shape=(train_3.shape[1],),name='in13', dtype='float32')            
    out13 = ks.layers.Dense(train_3.shape[1])(model_in13) 
#    out13 = Dropout(0.4)(out13)
    
    model_in14 = ks.Input(shape=(train_4.shape[1],),name='in14', dtype='float32')            
    out14 = ks.layers.Dense(train_4.shape[1])(model_in14) 
#    out14 = Dropout(0.4)(out14)
       
    #### Embedding Layers
    seq_title = Input(shape=[100], name="in18")
    seq_param = Input(shape=[20], name="in19")
    seq_desc = Input(shape=[100], name="in20")
            
    region = Input(shape=[1], name="in21")
    city = Input(shape=[1], name="in22")
    category_name = Input(shape=[1], name="in23")
    parent_category_name = Input(shape=[1], name="in24")
    ut = Input(shape=[1], name="in25")
    p1 = Input(shape=[1], name="in26")
    p2 = Input(shape=[1], name="in27")
    p3 = Input(shape=[1], name="in28")
#    uid = Input(shape = [1], name = 'in29')

    emb_seq_title = Embedding(100001, 100)(seq_title)
    emb_seq_param = Embedding(10001, 100)(seq_param)
    emb_seq_description = Embedding(100001, 100)(seq_desc)

    emb_region = Embedding(28, 10)(region)
    emb_city = Embedding(1752, 50)(city)
    emb_category_name = Embedding(47, 10)(category_name)
    emb_parent_category_name = Embedding(9, 3)(parent_category_name)
    emb_p1 = Embedding(372, 50)(p1)
    emb_p2 = Embedding(278, 50)(p2)
    emb_p3 = Embedding(1277, 50)(p3)
    emb_ut = Embedding(3, 1)(ut)
#    emb_uid = Embedding(1009909, 10)(uid)  

    rnn_layer1 = GRU(50, return_sequences=True) (emb_seq_title)          
    rnn_layer1 = Attention(50)(rnn_layer1)
    rnn_layer2 = GRU(20, return_sequences=True) (emb_seq_param)
    rnn_layer2 = Attention(20)(rnn_layer2)
    rnn_layer3 = GRU(50, return_sequences=True) (emb_seq_description)             
    rnn_layer3 = Attention(50)(rnn_layer3)    

#    too slow without GPU, hence leave it out
    #rnn_layer11 = Bidirectional(GRU(50, return_sequences=True)) (emb_seq_title)          
    #rnn_layer11 = Attention(50)(rnn_layer11)
    #rnn_layer21 = Bidirectional(GRU(20, return_sequences=True)) (emb_seq_param)
    #rnn_layer21 = Attention(20)(rnn_layer21)
    #rnn_layer31 = Bidirectional(GRU(50, return_sequences=True)) (emb_seq_description)             
    #rnn_layer31 = Attention(50)(rnn_layer31)    

    rnn_layer12 = GRU(50, return_sequences=True) (emb_seq_title)          
    rnn_layer12 = GRU(50)(rnn_layer12)
    rnn_layer22 = GRU(20, return_sequences=True) (emb_seq_param)
    rnn_layer22 = GRU(50)(rnn_layer22)
    rnn_layer32 = GRU(50, return_sequences=True) (emb_seq_description)             
    rnn_layer32 = GRU(50)(rnn_layer32)    

    #rnn_layer13 = Bidirectional(GRU(50, return_sequences=True)) (emb_seq_title)          
    #rnn_layer13 = Bidirectional(GRU(50))(rnn_layer13)
    #rnn_layer23 = Bidirectional(GRU(20, return_sequences=True)) (emb_seq_param)
    #rnn_layer23 = Bidirectional(GRU(50))(rnn_layer23)
    #rnn_layer33 = Bidirectional(GRU(50, return_sequences=True)) (emb_seq_description)             
    #rnn_layer33 = Bidirectional(GRU(50))(rnn_layer33)    

    main_l = concatenate([
        rnn_layer1, rnn_layer2, rnn_layer3
       #, rnn_layer11, rnn_layer21, rnn_layer31
        , rnn_layer12, rnn_layer22, rnn_layer32                    
        #, rnn_layer13, rnn_layer23, rnn_layer33                   
        , Flatten() (emb_region)
        , Flatten() (emb_city)
        , Flatten() (emb_category_name)
        , Flatten() (emb_parent_category_name)
        , Flatten() (emb_p1)
        , Flatten() (emb_p2)
        , Flatten() (emb_p3)
        , Flatten() (emb_ut)
        ,out1, out9,  
        out11, out12, out13, out14
    ])

    main_l = Dense(2000)(main_l)
    main_l = ks.layers.advanced_activations.PReLU(name='xx1')(main_l)
    main_l = Dropout(0.1)(main_l)
    main_l = Dense(1000)(main_l)
    main_l = ks.layers.advanced_activations.PReLU(name='xx2')(main_l)
    main_l = Dropout(0.1)(main_l)
   
    out = ks.layers.Dense(1, init='zero')(main_l)
    model = ks.models.Model([model_in1, model_in9,
    model_in11,model_in12,model_in13,model_in14,
    seq_title, seq_param, seq_desc, region, city, category_name, parent_category_name,
    ut, p1, p2, p3#, uid
    ], out)
    model.compile(loss = ["MSE"], metrics=[root_mean_squared_error], optimizer = ks.optimizers.Adam(lr=0.00003, epsilon=1e-08, decay=0., clipnorm = 0.85))
    return(model)

def nn_model4():
    model_in1 = ks.Input(shape=(train_df.shape[1],),name='in1', dtype='float32')
    out1 = ks.layers.Dense(train_df.shape[1])(model_in1)        
#    out1 = Dropout(0.4)(out1)

    model_in2 = ks.Input(shape=(param_train_tfidf.shape[1],),name='in2', dtype='float32', sparse=True)
    out2 = ks.layers.Dense(200)(model_in2)    
#    out2 = Dropout(0.4)(out2)

    model_in3 = ks.Input(shape=(title_train_tfidf.shape[1],),name='in3', dtype='float32', sparse=True)
    out3 = ks.layers.Dense(300)(model_in3)    #300#2243
#    out3 = Dropout(0.4)(out3)

    model_in4 = ks.Input(shape=(desc_train_tfidf.shape[1],),name='in4', dtype='float32', sparse=True)            
    out4 = ks.layers.Dense(200)(model_in4) 
#    out4 = Dropout(0.4)(out4)
    
    model_in5 = ks.Input(shape=(city_train.shape[1],),name='in5', dtype='float32', sparse=True)            
    out5 = ks.layers.Dense(city_train.shape[1])(model_in5) #
#    out5 = Dropout(0.4)(out5)

    model_in6 = ks.Input(shape=(pcn_train.shape[1],),name='in6', dtype='float32', sparse=True)            
    out6 = ks.layers.Dense(pcn_train.shape[1])(model_in6) 
#    out6 = Dropout(0.4)(out6)

    model_in7 = ks.Input(shape=(cn_train.shape[1],),name='in7', dtype='float32', sparse=True)            
    out7 = ks.layers.Dense(cn_train.shape[1])(model_in7) 
#    out7 = Dropout(0.4)(out7)

    model_in8 = ks.Input(shape=(ut_train.shape[1],),name='in8', dtype='float32', sparse=True)            
    out8 = ks.layers.Dense(3)(model_in8) 
#    out8 = Dropout(0.4)(out8)

    model_in9 = ks.Input(shape=(week_train.shape[1],),name='in9', dtype='float32', sparse=True)            
    out9 = ks.layers.Dense(7)(model_in9) 
#    out9 = Dropout(0.4)(out9)

    model_in10 = ks.Input(shape=(region_train.shape[1],),name='in10', dtype='float32', sparse=True)            
    out10 = ks.layers.Dense(region_train.shape[1])(model_in10) 
#    out10 = Dropout(0.4)(out10)

    model_in11 = ks.Input(shape=(train_x1.shape[1],),name='in11', dtype='float32')            
    out11 = ks.layers.Dense(train_x1.shape[1])(model_in11) 
#    out11 = Dropout(0.4)(out11)

    model_in12 = ks.Input(shape=(train_x2.shape[1],),name='in12', dtype='float32')            
    out12 = ks.layers.Dense(train_x2.shape[1])(model_in12) 
#    out12 = Dropout(0.4)(out12)

    model_in13 = ks.Input(shape=(train_3.shape[1],),name='in13', dtype='float32')            
    out13 = ks.layers.Dense(train_3.shape[1])(model_in13) 
#    out13 = Dropout(0.4)(out13)

    model_in14 = ks.Input(shape=(train_4.shape[1],),name='in14', dtype='float32')            
    out14 = ks.layers.Dense(train_4.shape[1])(model_in14) 
#    out14 = Dropout(0.4)(out14)

    model_in15 = ks.Input(shape=(tit_train_char.shape[1],),name='in15', dtype='float32', sparse=True)            
    out15 = ks.layers.Dense(200)(model_in15) 
#    out15 = Dropout(0.4)(out15)

    model_in16 = ks.Input(shape=(par_train_char.shape[1],),name='in16', dtype='float32', sparse=True)            
    out16 = ks.layers.Dense(200)(model_in16) 
#    out16 = Dropout(0.4)(out16)

    model_in17 = ks.Input(shape=(cat_train_char.shape[1],),name='in17', dtype='float32', sparse=True)            
    out17 = ks.layers.Dense(200)(model_in17) 
#    out17 = Dropout(0.4)(out17)
    
    x = concatenate([out1, out2, out3, out4, out5, out6, out7, out8, out9, out10, 
                     out11, out12, out13, out14, out15, out16, out17])    
    out = ks.layers.Dense(2000)(x)#1000-2219
    out = ks.layers.advanced_activations.PReLU(name='prelu5')(out)
    out = Dropout(0.1)(out)
    out = ks.layers.Dense(1000)(out)
    out = ks.layers.advanced_activations.PReLU(name='prelu51')(out)
    out = Dropout(0.1)(out)
   
    out = ks.layers.Dense(1, init='zero')(out)
    model = ks.models.Model([model_in1, model_in2, model_in3, model_in4, model_in5, model_in6, model_in7, model_in8, model_in9,
    model_in10,model_in11,model_in12,model_in13,model_in14,model_in15,model_in16,model_in17
    ], out)
    model.compile(loss = ["MSE"], metrics=[root_mean_squared_error], optimizer = ks.optimizers.Adam(lr=0.00003, epsilon=1e-08, decay=0.))
    return(model)

def root_mean_squared_error(y_true, y_pred):
        return K.sqrt(K.mean(K.square(y_pred - y_true))) 

i = 0
nbag = 1
nfold = 5
oobval = np.zeros((train_df.shape[0],2))
oobtest = np.zeros((test_df.shape[0],2))
valerr = []
val_scores = []

np.random.seed(2018)
for x in np.arange(nbag):
    for seed in [2018]:
        kf = model_selection.KFold(n_splits=nfold, shuffle=True, random_state=seed)    
        for dev_index, val_index in kf.split(y):
            dev_X, val_X = train_df.values[dev_index,:], train_df.values[val_index,:]
            dev_y, val_y = y[dev_index], y[val_index]
            param_train_tfidf_dev, param_train_tfidf_val = param_train_tfidf[dev_index,:], param_train_tfidf[val_index,:]
            title_train_tfidf_dev, title_train_tfidf_val =title_train_tfidf[dev_index,:], title_train_tfidf[val_index,:]
            desc_train_tfidf_dev, desc_train_tfidf_val =desc_train_tfidf[dev_index,:], desc_train_tfidf[val_index,:]
            region_dev, region_val = region_train[dev_index,:],region_train[val_index,:]
            pcn_dev, pcn_val = pcn_train[dev_index,:],  pcn_train[val_index,:]
            cn_dev, cn_val = cn_train[dev_index,:], cn_train[val_index,:]
            ut_dev, ut_val = ut_train[dev_index,:], ut_train[val_index,:]            
            city_dev, city_val = city_train[dev_index,:], city_train[val_index,:]            
            week_dev, week_val = week_train[dev_index,:], week_train[val_index,:]              
            dev_1, val_1 = train_x1.values[dev_index,:], train_x1.values[val_index,:]
            dev_2, val_2 = train_x2.values[dev_index,:], train_x2.values[val_index,:]
            dev_3, val_3 = train_3.values[dev_index,:], train_3.values[val_index,:]                                    
            dev_4, val_4 = train_4[dev_index,:], train_4[val_index,:]                                    
            tit_dev_char, tit_val_char = tit_train_char[dev_index,:], tit_train_char[val_index,:]
            par_dev_char, par_val_char = par_train_char[dev_index,:], par_train_char[val_index,:]
            cat_dev_char, cat_val_char = cat_train_char[dev_index,:], cat_train_char[val_index,:]
            
            ts_dev, ts_val = ts_tr[dev_index,:], ts_tr[val_index,:]
            pa_dev, pa_val = pa_tr[dev_index,:], pa_tr[val_index,:]
            desc_dev, desc_val = desc_tr[dev_index,:], desc_tr[val_index,:]
            
            re_dev, re_val = train_df2.region.values[dev_index],train_df2.region.values[val_index]
            ci_dev, ci_val = train_df2.city.values[dev_index],train_df2.city.values[val_index]
            ca_dev, ca_val = train_df2.category_name.values[dev_index],train_df2.category_name.values[val_index]
            pca_dev, pca_val = train_df2.parent_category_name.values[dev_index],train_df2.parent_category_name.values[val_index]
            utc_dev, utc_val = train_df2.user_type.values[dev_index],train_df2.user_type.values[val_index]
            p1_dev, p1_val = train_df2.param_1.values[dev_index],train_df2.param_1.values[val_index]
            p2_dev, p2_val = train_df2.param_2.values[dev_index],train_df2.param_2.values[val_index]
            p3_dev, p3_val = train_df2.param_3.values[dev_index],train_df2.param_3.values[val_index]          


            model = nn_model4()
            earlyStopping=EarlyStopping(monitor='val_loss', patience=1, verbose=1, mode='auto')
            checkpointer = ModelCheckpoint(filepath="./weights2XXLK.hdf5", verbose=1, save_best_only=True)

            fit = model.fit(get_keras_sparse(
                                            dev_X, 
                                            param_train_tfidf_dev, title_train_tfidf_dev, desc_train_tfidf_dev,
                                            city_dev,pcn_dev,cn_dev,ut_dev,week_dev, region_dev,
                                            dev_1, dev_2, dev_3, dev_4,
                                            tit_dev_char, par_dev_char, cat_dev_char,
                                            ts_dev, pa_dev, desc_dev,
                                            re_dev, ci_dev, ca_dev, pca_dev, utc_dev, p1_dev, p2_dev, p3_dev#, uid_dev
                                            ), dev_y, batch_size=4096, epochs = 4,#10
                                            
                                            validation_data=(
                                                    get_keras_sparse(val_X, 
                                                    param_train_tfidf_val, title_train_tfidf_val, desc_train_tfidf_val,
                                                    city_val,pcn_val,cn_val,ut_val,week_val, region_val,
                                                    val_1, val_2, val_3, val_4,
                                                    tit_val_char, par_val_char, cat_val_char,
                                                    ts_val, pa_val, desc_val,
                                                    re_val, ci_val, ca_val, pca_val, utc_val, p1_val, p2_val, p3_val#, uid_val
                                                    ), val_y), 
                                            verbose = 1,
                                            callbacks=[earlyStopping,checkpointer])

            model =load_model("./weights2XXLK.hdf5",custom_objects={'root_mean_squared_error': root_mean_squared_error})

            preds = model.predict(get_keras_sparse(val_X, 
                                                    param_train_tfidf_val, title_train_tfidf_val, desc_train_tfidf_val,
                                                    city_val,pcn_val,cn_val,ut_val,week_val, region_val,
                                                    val_1, val_2, val_3, val_4,
                                                    tit_val_char, par_val_char, cat_val_char,
                                                    ts_val, pa_val, desc_val,
                                                    re_val, ci_val, ca_val, pca_val, utc_val, p1_val, p2_val, p3_val
                                                    )).reshape(-1,1)

            tstpreds = model.predict(get_keras_sparse(test_df.values, 
                                                    param_test_tfidf, title_test_tfidf, desc_test_tfidf,
                                                    city_test,pcn_test,cn_test,ut_test,week_test, region_test,
                                                    test_x1.values, test_x2.values, test_3.values, test_4,
                                                    tit_test_char, par_test_char, cat_test_char,
                                                    ts_te, pa_te, desc_te,
                                                    test_df2.region.values, 
                                                    test_df2.city.values, 
                                                    test_df2.category_name.values, 
                                                    test_df2.parent_category_name.values, 
                                                    test_df2.user_type.values, 
                                                    test_df2.param_1.values, 
                                                    test_df2.param_2.values, 
                                                    test_df2.param_3.values
                                                    )).reshape(-1,1)
            print("predicting..")
            oobval[val_index, 0:1] += preds
            oobtest[:, 0:1] += tstpreds
            valerr.append(mean_squared_error(val_y, preds)**0.5)
            print(valerr, "mean:", np.mean(valerr), "std:", np.std(valerr))

            model = nn_model5()
            earlyStopping=EarlyStopping(monitor='val_loss', patience=1, verbose=1, mode='auto')
            checkpointer = ModelCheckpoint(filepath="./weights3XXLK.hdf5", verbose=1, save_best_only=True)

            fit = model.fit(get_keras_sparse2(
                                            dev_X, 
                                            week_dev,
                                            dev_1, dev_2, dev_3, dev_4,
                                            ts_dev, pa_dev, desc_dev,
                                            re_dev, ci_dev, ca_dev, pca_dev, utc_dev, p1_dev, p2_dev, p3_dev#, uid_dev
                                            ), dev_y, batch_size=4096, epochs = 3,#10
                                            
                                            validation_data=(
                                                    get_keras_sparse2(val_X, 
                                                    week_val,
                                                    val_1, val_2, val_3, val_4,
                                                    ts_val, pa_val, desc_val,
                                                    re_val, ci_val, ca_val, pca_val, utc_val, p1_val, p2_val, p3_val#, uid_val
                                                    ), val_y), 
                                            verbose = 1,
                                            callbacks=[earlyStopping,checkpointer])

            model.load_weights("./weights3XXLK.hdf5")

            preds = model.predict(get_keras_sparse2(val_X, 
                                                    week_val, 
                                                    val_1, val_2, val_3, val_4,
                                                    ts_val, pa_val, desc_val,
                                                    re_val, ci_val, ca_val, pca_val, utc_val, p1_val, p2_val, p3_val#, uid_val
                                                    )).reshape(-1,1)

            tstpreds = model.predict(get_keras_sparse2(test_df.values, 
                                                    week_test,
                                                    test_x1.values, test_x2.values, test_3.values, test_4,
                                                    ts_te, pa_te, desc_te,
                                                    test_df2.region.values, 
                                                    test_df2.city.values, 
                                                    test_df2.category_name.values, 
                                                    test_df2.parent_category_name.values, 
                                                    test_df2.user_type.values, 
                                                    test_df2.param_1.values, 
                                                    test_df2.param_2.values, 
                                                    test_df2.param_3.values
                                                    )).reshape(-1,1)
            print("predicting..")
            oobval[val_index, 1:2] += preds
            oobtest[:, 1:2] += tstpreds
            valerr.append(mean_squared_error(val_y, preds)**0.5)
            print(valerr, "mean:", np.mean(valerr), "std:", np.std(valerr))

            del(val_X, 
                param_train_tfidf_val, title_train_tfidf_val, desc_train_tfidf_val,
                city_val,pcn_val,cn_val,ut_val,week_val, region_val,
                val_1, val_2, val_3, val_4,
                tit_val_char, par_val_char, cat_val_char,
                ts_val, pa_val, desc_val,
                re_val, ci_val, ca_val, pca_val, utc_val, p1_val, p2_val, p3_val,
                dev_X, 
                param_train_tfidf_dev, title_train_tfidf_dev, desc_train_tfidf_dev,
                city_dev,pcn_dev,cn_dev,ut_dev,week_dev, region_dev,
                dev_1, dev_2, dev_3, dev_4,
                tit_dev_char, par_dev_char, cat_dev_char,
                re_dev, ci_dev, ca_dev, pca_dev, utc_dev, p1_dev, p2_dev, p3_dev                
                );gc.collect()
            
tstpred = oobtest / (nfold * nbag)#*3
oobpred = oobval / (nbag)    #*3)
oobpred[oobpred>1] = 1
oobpred[oobpred<0] = 0

tstpred[tstpred>1] = 1
tstpred[tstpred<0] = 0

joblib.dump(oobpred,'../input/train_keras_all.pkl')
joblib.dump(tstpred,'../input/test_keras_all.pkl')

#####################

# Making a submission file #
tstpreds = (tstpred[:,0]+tstpred[:,1])/2
sub_df = pd.DataFrame({"item_id":test_id})
sub_df["deal_probability"] = tstpreds
sub_df.to_csv("../output/keras1.csv", index=False)    
#PLB -2220
# Making a submission file #
tstpred1 = (tstpred[:,0])/1
sub_df = pd.DataFrame({"item_id":test_id})
sub_df["deal_probability"] = tstpred1
sub_df.to_csv("../output/keras2.csv", index=False)    
#plb 2222
# Making a submission file #
tstpred = (tstpred[:,1])/1
sub_df = pd.DataFrame({"item_id":test_id})
sub_df["deal_probability"] = tstpred
sub_df.to_csv("../output/keras3.csv", index=False)    
# plb 2223


