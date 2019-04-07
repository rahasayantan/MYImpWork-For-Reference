# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from tqdm import tqdm
tqdm.pandas()
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn import metrics

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import *
from keras.models import *
from keras import initializers, regularizers, constraints, optimizers, layers
from keras.initializers import *
from keras.optimizers import *
import keras.backend as K
from keras.callbacks import *
import tensorflow as tf
import os
import time
import gc
import re
from unidecode import unidecode
import os
import operator 
import nltk

print(os.listdir("../input"))
import re
puncts = [',', '.', '"', ':', ')', '(', '-', '!', '?', '|', ';', "'", '$', '&', '/', '[', ']', '>', '%', '=', '#', '*', '+', '\\', '•',  '~', '@', '£', 
 '·', '_', '{', '}', '©', '^', '®', '`',  '<', '→', '°', '€', '™', '›',  '♥', '←', '×', '§', '″', '′', 'Â', '█', '½', 'à', '…', 
 '“', '★', '”', '–', '●', 'â', '►', '−', '¢', '²', '¬', '░', '¶', '↑', '±', '¿', '▾', '═', '¦', '║', '―', '¥', '▓', '—', '‹', '─', 
 '▒', '：', '¼', '⊕', '▼', '▪', '†', '■', '’', '▀', '¨', '▄', '♫', '☆', 'é', '¯', '♦', '¤', '▲', 'è', '¸', '¾', 'Ã', '⋅', '‘', '∞', 
 '∙', '）', '↓', '、', '│', '（', '»', '，', '♪', '╩', '╚', '³', '・', '╦', '╣', '╔', '╗', '▬', '❤', 'ï', 'Ø', '¹', '≤', '‡', '√', ]

def clean_text(x):
    x = str(x)
    for punct in puncts:
        x = x.replace(punct, f' {punct} ')
    return x

def clean_numbers(x):
    x = re.sub('[0-9]{5,}', '#####', x)
    x = re.sub('[0-9]{4}', '####', x)
    x = re.sub('[0-9]{3}', '###', x)
    x = re.sub('[0-9]{2}', '##', x)
    return x
    
mispell_dict = {'cryptocurrencies':'cryptocurrency',
'Quorans':'Quora sucscribers', 
'colour': 'color', 
'centre': 'center', 
'favourite': 'favorite', 
'travelling': 'traveling', 
'counselling': 'counseling', 
'theatre': 'theater', 
'cancelled': 'canceled', 
'labour': 'labor', 
'organisation': 'organization', 
'wwii': 'world war 2', 
'citicise': 'criticize', 
'youtu ': 'youtube ', 
'Qoura': 'Quora', 
'sallary': 'salary', 
'Whta': 'What', 
'narcisist': 'narcissist', 
'howdo': 'how do', 
'whatare': 'what are', 
'howcan': 'how can', 
'howmuch': 'how much', 
'howmany': 'how many', 
'whydo': 'why do', 
'doI': 'do I', 
'theBest': 'the best', 
'howdoes': 'how does', 
'mastrubation': 'masturbation', 
'mastrubate': 'masturbate', 
"mastrubating": 'masturbating', 
'pennis': 'penis', 
'Etherium': 'Ethereum', 
'narcissit': 'narcissist', 
'bigdata': 'big data', 
'2k17': '2017', 
'2k18': '2018', 
'qouta': 'quota', 
'exboyfriend': 'ex boyfriend', 
'airhostess': 'air hostess', 
"whst": 'what', 
'watsapp': 'whatsapp', 
'demonitisation': 'demonetization', 
'demonitization': 'demonetization', 
'demonetisation': 'demonetization',
"tryin'":"trying",
"aren't" : "are not",
"can't" : "cannot",
"couldn't" : "could not",
"didn't" : "did not",
"doesn't" : "does not",
"don't" : "do not",
"hadn't" : "had not",
"hasn't" : "has not",
"haven't" : "have not",
"he'd" : "he would",
"he'll" : "he will",
"he's" : "he is",
"i'd" : "I would",
"i'd" : "I had",
"i'll" : "I will",
"i'm" : "I am",
"isn't" : "is not",
"it's" : "it is",
"it'll":"it will",
"i've" : "I have",
"let's" : "let us",
"mightn't" : "might not",
"mustn't" : "must not",
"shan't" : "shall not",
"she'd" : "she would",
"she'll" : "she will",
"she's" : "she is",
"shouldn't" : "should not",
"that's" : "that is",
"there's" : "there is",
"they'd" : "they would",
"they'll" : "they will",
"they're" : "they are",
"they've" : "they have",
"we'd" : "we would",
"we're" : "we are",
"weren't" : "were not",
"we've" : "we have",
"what'll" : "what will",
"what're" : "what are",
"what's" : "what is",
"what've" : "what have",
"where's" : "where is",
"who'd" : "who would",
"who'll" : "who will",
"who're" : "who are",
"who's" : "who is",
"who've" : "who have",
"won't" : "will not",
"wouldn't" : "would not",
"you'd" : "you would",
"you'll" : "you will",
"you're" : "you are",
"you've" : "you have",
"'re": " are",
"wasn't": "was not",
"we'll":" will",
"didn't": "did not",
"tryin'":"trying"
}

def _get_mispell(mispell_dict):
    mispell_re = re.compile('(%s)' % '|'.join(mispell_dict.keys()))
    return mispell_dict, mispell_re

mispellings, mispellings_re = _get_mispell(mispell_dict)
def replace_typical_misspell(text):
    def replace(match):
        return mispellings[match.group(0)]
    return mispellings_re.sub(replace, text)


## data clean
train_df = pd.read_csv("../input/train.csv")
test_df = pd.read_csv("../input/test.csv")

print("Train shape : ",train_df.shape)
print("Test shape : ",test_df.shape)

# Clean the text
train_df["question_text"] = train_df["question_text"].progress_apply(lambda x: clean_text(x))
test_df["question_text"] = test_df["question_text"].progress_apply(lambda x: clean_text(x))

# Clean numbers
train_df["question_text"] = train_df["question_text"].progress_apply(lambda x: clean_numbers(x))
test_df["question_text"] = test_df["question_text"].progress_apply(lambda x: clean_numbers(x))

# Clean speelings
train_df["question_text"] = train_df["question_text"].progress_apply(lambda x: replace_typical_misspell(x))
test_df["question_text"] = test_df["question_text"].progress_apply(lambda x: replace_typical_misspell(x))

## fill up the missing values
train_X = train_df["question_text"].fillna("_##_").values
test_X = test_df["question_text"].fillna("_##_").values

# lower
train_df["question_text_lw"] = train_df["question_text"].progress_apply(lambda x: x.lower())
test_df["question_text_lw"] = test_df["question_text"].progress_apply(lambda x: x.lower())

train_X2 = train_df["question_text_lw"].fillna("_##_").values
test_X2 = test_df["question_text_lw"].fillna("_##_").values

## Tokenize the sentences - Glove
max_features = 200000
tokenizer = Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(list(train_X))
train_X = tokenizer.texts_to_sequences(train_X)
test_X = tokenizer.texts_to_sequences(test_X)
word_index1 = tokenizer.word_index
print(len(word_index1))

tokenizer2 = Tokenizer(num_words=max_features)
tokenizer2.fit_on_texts(list(train_X2))
train_X2 = tokenizer.texts_to_sequences(train_X2)
test_X2 = tokenizer.texts_to_sequences(test_X2)
word_index2 = tokenizer2.word_index
print(len(word_index2))

## Pad the sentences 
maxlen = 72 # from other kernels
train_XX1 = pad_sequences(train_X, maxlen=maxlen)
test_XX1 = pad_sequences(test_X, maxlen=maxlen)
train_XX2 = pad_sequences(train_X2, maxlen=maxlen)
test_XX2 = pad_sequences(test_X2, maxlen=maxlen)
## Get the target values
train_y = train_df['target'].values
del(train_X, test_X, train_X2, test_X2); gc.collect()

def load_glove(word_index):
    EMBEDDING_FILE = '../input/embeddings/glove.840B.300d/glove.840B.300d.txt'
    def get_coefs(word,*arr): return word, np.asarray(arr, dtype='float32')
    embeddings_index = dict(get_coefs(*o.split(" ")) for o in open(EMBEDDING_FILE))

    all_embs = np.stack(embeddings_index.values())
    emb_mean,emb_std = all_embs.mean(), all_embs.std()
    embed_size = all_embs.shape[1]

    # word_index = tokenizer.word_index
    nb_words = min(max_features, len(word_index))
    embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words+1, embed_size))
    for word, i in word_index.items():
        if i >= max_features: continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None: embedding_matrix[i] = embedding_vector
            
    return embedding_matrix 

def load_para(word_index):
    EMBEDDING_FILE = '../input/embeddings/paragram_300_sl999/paragram_300_sl999.txt'
    def get_coefs(word,*arr): return word, np.asarray(arr, dtype='float32')
    embeddings_index = dict(get_coefs(*o.split(" ")) for o in open(EMBEDDING_FILE, encoding="utf8", errors='ignore') if len(o)>100)

    all_embs = np.stack(embeddings_index.values())
    emb_mean,emb_std = all_embs.mean(), all_embs.std()
    embed_size = all_embs.shape[1]

    # word_index = tokenizer.word_index
    nb_words = min(max_features, len(word_index))
    embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words+1, embed_size))
    for word, i in word_index.items():
        if i >= max_features: continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None: embedding_matrix[i] = embedding_vector
    
    return embedding_matrix
    
start_time = time.time()

embedding_matrix_1 = load_glove(word_index1)
embedding_matrix_2 = load_para(word_index1)

total_time = (time.time() - start_time) / 60
print("Took {:.2f} minutes".format(total_time))

# I will see if it makes sense to use the two embeddings seperately
#embedding_matrix = np.mean([embedding_matrix_1, embedding_matrix_2], axis=0)
#print(np.shape(embedding_matrix))

#del embedding_matrix_1, embedding_matrix_2
#gc.collect()


import keras.backend as K
from keras.layers import Lambda, Activation, Dropout, Embedding, SpatialDropout1D, Dense, merge, concatenate
from keras.layers import TimeDistributed  # This applies the model to every timestep in the input sequences
from keras.layers import Bidirectional, LSTM
from keras.layers import Dense, Input, CuDNNLSTM, Embedding, Dropout, Activation, CuDNNGRU, Conv1D
# from keras.layers.advanced_activations import ELU
from keras.models import Sequential
from keras import regularizers
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import GlobalAveragePooling1D, GlobalMaxPooling1D

class Attention(Layer):
    def __init__(self, step_dim,
                 W_regularizer=None, b_regularizer=None,
                 W_constraint=None, b_constraint=None,
                 bias=True, **kwargs):
        self.supports_masking = True
        self.init = initializers.get('glorot_uniform')

        self.W_regularizer = regularizers.get(W_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        self.step_dim = step_dim
        self.features_dim = 0
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3

        self.W = self.add_weight((input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        self.features_dim = input_shape[-1]

        if self.bias:
            self.b = self.add_weight((input_shape[1],),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)
        else:
            self.b = None

        self.built = True

    def compute_mask(self, input, input_mask=None):
        return None

    def call(self, x, mask=None):
        features_dim = self.features_dim
        step_dim = self.step_dim

        eij = K.reshape(K.dot(K.reshape(x, (-1, features_dim)),
                        K.reshape(self.W, (features_dim, 1))), (-1, step_dim))

        if self.bias:
            eij += self.b

        eij = K.tanh(eij)

        a = K.exp(eij)

        if mask is not None:
            a *= K.cast(mask, K.floatx())

        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())

        a = K.expand_dims(a)
        weighted_input = x * a
        return K.sum(weighted_input, axis=1)

    def compute_output_shape(self, input_shape):
        return input_shape[0],  self.features_dim

def lstmmodel():
    inp = Input(shape=(maxlen,))
    x = Embedding(embedding_matrix_1.shape[0], embedding_matrix_1.shape[1], 
                    weights=[embedding_matrix_1], trainable=False)(inp)
    x =  SpatialDropout1D(.1)(x)
    x = Bidirectional(CuDNNLSTM(128, return_sequences=True))(x)
    #x = Bidirectional(CuDNNLSTM(128, return_sequences=True))(x)
    x = Bidirectional(CuDNNLSTM(64, return_sequences=True))(x)
    x1 = Attention(maxlen)(x)
    x2 = GlobalAveragePooling1D()(x)
    x3 = GlobalMaxPooling1D()(x)
    x = concatenate([x1, x2, x3])
    x = Dense(200, activation="relu")(x)
    x = BatchNormalization()(x)
    x = Dropout(.1)(x)
    x = Dense(200, activation="relu")(x)
    x = BatchNormalization()(x)
    x = Dropout(.1)(x)
    x = Dense(1, activation="sigmoid")(x)
    model = Model(inputs=inp, outputs=x)
    model.compile(loss='binary_crossentropy', optimizer=Adam(lr=1e-2), metrics=['accuracy'])
    return model

def f1_smart(y_true, y_pred):
    args = np.argsort(y_pred)
    tp = y_true.sum()
    fs = (tp - np.cumsum(y_true[args[:-1]])) / np.arange(y_true.shape[0] + tp - 1, tp, -1)
    res_idx = np.argmax(fs)
    return 2 * fs[res_idx], (y_pred[args[res_idx]] + y_pred[args[res_idx + 1]]) / 2
    
#Start training

kfold = StratifiedKFold(n_splits=5, random_state=10, shuffle=True)
bestscore = []
y_test = np.zeros((test_XX1.shape[0] ))
np.random.seed(2018)
for i, (train_index, valid_index) in enumerate(kfold.split(train_XX1, train_y)):
    X_train, X_val, Y_train, Y_val = (train_XX1[train_index], train_XX1[valid_index], 
                                        train_y[train_index], train_y[valid_index])
    filepath="weights_best.h5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=2, save_best_only=True, mode='min')
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=1, min_lr=0.0001, verbose=2)
    earlystopping = EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=2, verbose=2, mode='auto')
    callbacks = [checkpoint, reduce_lr]
    model = lstmmodel()
    if i == 0:print(model.summary()) 
    model.fit(X_train, Y_train, batch_size=256, epochs=10, validation_data=(X_val, Y_val), 
    verbose=2, callbacks=callbacks)
    model.load_weights(filepath)
    y_pred = model.predict([X_val], batch_size=1024, verbose=2)
    y_test += np.squeeze(model.predict([test_XX1], batch_size=1024, verbose=1))/5
    f1, threshold = f1_smart(np.squeeze(Y_val), np.squeeze(y_pred))
    print('Optimal F1: {:.4f} at threshold: {:.4f}'.format(f1, threshold))
    bestscore.append(threshold)

sub = test_df[['qid']]
y_test = y_test.reshape((-1, 1))
pred_test_y = (y_test>np.mean(bestscore)).astype(int)
sub['prediction'] = pred_test_y
sub.to_csv("submission.csv", index=False)