# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np
np.random.seed(42)
import pandas as pd
import string
import re

import gensim
from collections import Counter
import pickle

import tensorflow as tf
from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

from keras.models import Model
from keras.layers import Input, Dense, Dropout, Conv1D, Embedding, SpatialDropout1D, concatenate
from keras.layers import GRU, LSTM,Bidirectional, GlobalAveragePooling1D, GlobalMaxPooling1D
from keras.layers import CuDNNLSTM, CuDNNGRU
from keras.preprocessing import text, sequence

from keras.callbacks import Callback
from keras import optimizers
from keras.layers import Lambda

import warnings
warnings.filterwarnings('ignore')

from nltk.corpus import stopwords

import os
os.environ['OMP_NUM_THREADS'] = '4'

import gc
from keras import backend as K
from sklearn.model_selection import KFold

from unidecode import unidecode

import time

eng_stopwords = set(stopwords.words("english"))

# 1. preprocessing
train = pd.read_csv("../input/processing-helps-boosting-about-0-0005-on-lb/train_processed.csv")
test = pd.read_csv("../input/processing-helps-boosting-about-0-0005-on-lb/test_processed.csv")



# 2. remove numbers
def clean_numbers(x):
    x = re.sub('[0-9]{5,}', '#####', x)
    x = re.sub('[0-9]{4}', '####', x)
    x = re.sub('[0-9]{3}', '###', x)
    x = re.sub('[0-9]{2}', '##', x)
    return x



#3.  remove non-ascii

special_character_removal = re.compile(r'[^A-Za-z\.\-\?\!\,\#\@\% ]',re.IGNORECASE)
def clean_text(x):
    x_ascii = unidecode(x)
    x_clean = special_character_removal.sub('',x_ascii)
    return x_clean

train['clean_text'] = train['comment_text'].apply(lambda x: clean_text(str(x)))
test['clean_text'] = test['comment_text'].apply(lambda x: clean_text(str(x)))

X_train = train['clean_text'].fillna("something").values
y_train = train[["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]].values
X_test = test['clean_text'].fillna("something").values


def add_features(df):
    
    df['comment_text'] = df['comment_text'].apply(lambda x:str(x))
    df['total_length'] = df['comment_text'].apply(len)
    df['capitals'] = df['comment_text'].apply(lambda comment: sum(1 for c in comment if c.isupper()))
    df['caps_vs_length'] = df.apply(lambda row: float(row['capitals'])/float(row['total_length']),
                                axis=1)
    df['num_words'] = df.comment_text.str.count('\S+')
    df['num_unique_words'] = df['comment_text'].apply(lambda comment: len(set(w for w in comment.split())))
    df['words_vs_unique'] = df['num_unique_words'] / df['num_words']  

    return df

train = add_features(train)
test = add_features(test)

features = train[['caps_vs_length', 'words_vs_unique']].fillna(0)
test_features = test[['caps_vs_length', 'words_vs_unique']].fillna(0)

ss = StandardScaler()
ss.fit(np.vstack((features, test_features)))
features = ss.transform(features)
test_features = ss.transform(test_features)

# For best score (Public: 9869, Private: 9865), change to max_features = 283759, maxlen = 900
max_features = 10000
maxlen = 50

tokenizer = text.Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(list(X_train) + list(X_test))
X_train_sequence = tokenizer.texts_to_sequences(X_train)
X_test_sequence = tokenizer.texts_to_sequences(X_test)

x_train = sequence.pad_sequences(X_train_sequence, maxlen=maxlen)
x_test = sequence.pad_sequences(X_test_sequence, maxlen=maxlen)
print(len(tokenizer.word_index))

# Load the FastText Web Crawl vectors
EMBEDDING_FILE_FASTTEXT="../input/fasttext-crawl-300d-2m/crawl-300d-2M.vec"
EMBEDDING_FILE_TWITTER="../input/glove-twitter-27b-200d-txt/glove.twitter.27B.200d.txt"
def get_coefs(word, *arr): return word, np.asarray(arr, dtype='float32')
embeddings_index_ft = dict(get_coefs(*o.rstrip().rsplit(' ')) for o in open(EMBEDDING_FILE_FASTTEXT,encoding='utf-8'))
embeddings_index_tw = dict(get_coefs(*o.strip().split()) for o in open(EMBEDDING_FILE_TWITTER,encoding='utf-8'))

spell_model = gensim.models.KeyedVectors.load_word2vec_format(EMBEDDING_FILE_FASTTEXT)

# This code is  based on: Spellchecker using Word2vec by CPMP
# https://www.kaggle.com/cpmpml/spell-checker-using-word2vec

words = spell_model.index2word

w_rank = {}
for i,word in enumerate(words):
    w_rank[word] = i

WORDS = w_rank

# Use fast text as vocabulary
def words(text): return re.findall(r'\w+', text.lower())

def P(word): 
    "Probability of `word`."
    # use inverse of rank as proxy
    # returns 0 if the word isn't in the dictionary
    return - WORDS.get(word, 0)

def correction(word): 
    "Most probable spelling correction for word."
    return max(candidates(word), key=P)

def candidates(word): 
    "Generate possible spelling corrections for word."
    return (known([word]) or known(edits1(word)) or known(edits2(word)) or [word])

def known(words): 
    "The subset of `words` that appear in the dictionary of WORDS."
    return set(w for w in words if w in WORDS)

def edits1(word):
    "All edits that are one edit away from `word`."
    letters    = 'abcdefghijklmnopqrstuvwxyz'
    splits     = [(word[:i], word[i:])    for i in range(len(word) + 1)]
    deletes    = [L + R[1:]               for L, R in splits if R]
    transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R)>1]
    replaces   = [L + c + R[1:]           for L, R in splits if R for c in letters]
    inserts    = [L + c + R               for L, R in splits for c in letters]
    return set(deletes + transposes + replaces + inserts)

def edits2(word): 
    "All edits that are two edits away from `word`."
    return (e2 for e1 in edits1(word) for e2 in edits1(e1))

def singlify(word):
    return "".join([letter for i,letter in enumerate(word) if i == 0 or letter != word[i-1]])
    
word_index = tokenizer.word_index
nb_words = min(max_features, len(word_index))
embedding_matrix = np.zeros((nb_words,501))

something_tw = embeddings_index_tw.get("something")
something_ft = embeddings_index_ft.get("something")

something = np.zeros((501,))
something[:300,] = something_ft
something[300:500,] = something_tw
something[500,] = 0

def all_caps(word):
    return len(word) > 1 and word.isupper()

def embed_word(embedding_matrix,i,word):
    embedding_vector_ft = embeddings_index_ft.get(word)
    if embedding_vector_ft is not None: 
        if all_caps(word):
            last_value = np.array([1])
        else:
            last_value = np.array([0])
        embedding_matrix[i,:300] = embedding_vector_ft
        embedding_matrix[i,500] = last_value
        embedding_vector_tw = embeddings_index_tw.get(word)
        if embedding_vector_tw is not None:
            embedding_matrix[i,300:500] = embedding_vector_tw

            
# Fasttext vector is used by itself if there is no glove vector but not the other way around.
for word, i in word_index.items():
    
    if i >= max_features: continue
        
    if embeddings_index_ft.get(word) is not None:
        embed_word(embedding_matrix,i,word)
    else:
        # change to > 20 for better score.
        if len(word) > 0:
            embedding_matrix[i] = something
        else:
            word2 = correction(word)
            if embeddings_index_ft.get(word2) is not None:
                embed_word(embedding_matrix,i,word2)
            else:
                word2 = correction(singlify(word))
                if embeddings_index_ft.get(word2) is not None:
                    embed_word(embedding_matrix,i,word2)
                else:
                    embedding_matrix[i] = something     
                    
class RocAucEvaluation(Callback):
    def __init__(self, validation_data=(), interval=1):
        super(Callback, self).__init__()

        self.interval = interval
        self.X_val, self.y_val = validation_data
        self.max_score = 0
        self.not_better_count = 0

    def on_epoch_end(self, epoch, logs={}):
        if epoch % self.interval == 0:
            y_pred = self.model.predict(self.X_val, verbose=1)
            score = roc_auc_score(self.y_val, y_pred)
            print("\n ROC-AUC - epoch: %d - score: %.6f \n" % (epoch+1, score))
            if (score > self.max_score):
                print("*** New High Score (previous: %.6f) \n" % self.max_score)
                model.save_weights("best_weights.h5")
                self.max_score=score
                self.not_better_count = 0
            else:
                self.not_better_count += 1
                if self.not_better_count > 3:
                    print("Epoch %05d: early stopping, high score = %.6f" % (epoch,self.max_score))
                    self.model.stop_training = True
                    
def get_model(features,clipvalue=1.,num_filters=40,dropout=0.5,embed_size=501):
    features_input = Input(shape=(features.shape[1],))
    inp = Input(shape=(maxlen, ))
    
    # Layer 1: concatenated fasttext and glove twitter embeddings.
    x = Embedding(max_features, embed_size, weights=[embedding_matrix], trainable=False)(inp)
    
    # Uncomment for best result
    # Layer 2: SpatialDropout1D(0.5)
    #x = SpatialDropout1D(dropout)(x)
    
    # Uncomment for best result
    # Layer 3: Bidirectional CuDNNLSTM
    #x = Bidirectional(LSTM(num_filters, return_sequences=True))(x)


    # Layer 4: Bidirectional CuDNNGRU
    x, x_h, x_c = Bidirectional(GRU(num_filters, return_sequences=True, return_state = True))(x)  
    
    # Layer 5: A concatenation of the last state, maximum pool, average pool and 
    # two features: "Unique words rate" and "Rate of all-caps words"
    avg_pool = GlobalAveragePooling1D()(x)
    max_pool = GlobalMaxPooling1D()(x)
    
    x = concatenate([avg_pool, x_h, max_pool,features_input])
    
    # Layer 6: output dense layer.
    outp = Dense(6, activation="sigmoid")(x)

    model = Model(inputs=[inp,features_input], outputs=outp)
    adam = optimizers.adam(clipvalue=clipvalue)
    model.compile(loss='binary_crossentropy',
                  optimizer=adam,
                  metrics=['accuracy'])
    return model
    

model = get_model(features)

batch_size = 32

# Used epochs=100 with early exiting for best score.
epochs = 1
gc.collect()
K.clear_session()

# Change to 10
num_folds = 2 #number of folds

predict = np.zeros((test.shape[0],6))

# Uncomment for out-of-fold predictions
#scores = []
#oof_predict = np.zeros((train.shape[0],6))

kf = KFold(n_splits=num_folds, shuffle=True, random_state=239)

for train_index, test_index in kf.split(x_train):
    
    kfold_y_train,kfold_y_test = y_train[train_index], y_train[test_index]
    kfold_X_train = x_train[train_index]
    kfold_X_features = features[train_index]
    kfold_X_valid = x_train[test_index]
    kfold_X_valid_features = features[test_index] 
    
    gc.collect()
    K.clear_session()
    
    model = get_model(features)
    
    ra_val = RocAucEvaluation(validation_data=([kfold_X_valid,kfold_X_valid_features], kfold_y_test), interval = 1)
    
    model.fit([kfold_X_train,kfold_X_features], kfold_y_train, batch_size=batch_size, epochs=epochs, verbose=1,
             callbacks = [ra_val])
    gc.collect()
    
    #model.load_weights(bst_model_path)
    model.load_weights("best_weights.h5")
    
    predict += model.predict([x_test,test_features], batch_size=batch_size,verbose=1) / num_folds
    
    #gc.collect()
    # uncomment for out of fold predictions
    #oof_predict[test_index] = model.predict([kfold_X_valid, kfold_X_valid_features],batch_size=batch_size, verbose=1)
    #cv_score = roc_auc_score(kfold_y_test, oof_predict[test_index])
    
    #scores.append(cv_score)
    #print('score: ',cv_score)

print("Done")
#print('Total CV score is {}'.format(np.mean(scores)))    


sample_submission = pd.read_csv("../input/jigsaw-toxic-comment-classification-challenge/sample_submission.csv")
class_names = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
sample_submission[class_names] = predict
sample_submission.to_csv('model_9872_baseline_submission.csv',index=False)

# uncomment for out of fold predictions
#oof = pd.DataFrame.from_dict({'id': train['id']})
#for c in class_names:
#    oof[c] = np.zeros(len(train))
#    
#oof[class_names] = oof_predict
#for c in class_names:
#    oof['prediction_' +c] = oof[c]
#oof.to_csv('oof-model_9872_baseline_submission.csv', index=False)



train["question_text"] = train["question_text"].str.lower()
test["question_text"] = test["question_text"].str.lower()

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


train["question_text"] = train["question_text"].apply(lambda x: clean_text(x))
test["question_text"] = test["question_text"].apply(lambda x: clean_text(x))

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
    #x = Bidirectional(CuDNNLSTM(64, return_sequences=True))(x)
    #x = Bidirectional(CuDNNGRU(64, return_sequences=True))(x)
    #x = Bidirectional(CuDNNGRU(64, return_sequences=True))(x)
    #x = Attention(maxlen)(x)

    x1 = Attention(maxlen)(x)
    x2 = GlobalAveragePooling1D()(x)
    x3 = GlobalMaxPooling1D()(x)
    x = concatenate([x1, x2, x3])
    x = Dropout(.1)(x)
    x = Dense(64, activation="relu")(x)
    x = BatchNormalization()(x)
    x = Dropout(.1)(x)
    x = Dense(64, activation="relu")(x)
    x = BatchNormalization()(x)
    x = Dropout(.1)(x)
    x = Dense(1, activation="sigmoid")(x)
    model = Model(inputs=[inp], outputs=x)
    model.compile(loss='binary_crossentropy', optimizer=Adam(lr=1e-3, clipnorm = 0.85), metrics=['accuracy'])
    return model

def f1_smart(y_true, y_pred):
    args = np.argsort(y_pred)
    tp = y_true.sum()
    fs = (tp - np.cumsum(y_true[args[:-1]])) / np.arange(y_true.shape[0] + tp - 1, tp, -1)
    res_idx = np.argmax(fs)
    return 2 * fs[res_idx], (y_pred[args[res_idx]] + y_pred[args[res_idx + 1]]) / 2
    
#Start training
sub = test_df[['qid']]

np.random.seed(2018)
kfold = StratifiedKFold(n_splits=5, random_state=2018)#, shuffle=True)
bestscore = []
y_test = np.zeros((test_XX1.shape[0] ))
y_test_whole = np.zeros((test_XX1.shape[0], 5))

for i, (train_index, valid_index) in enumerate(kfold.split(train_XX1, train_y)):
    X_train, X_val, Y_train, Y_val = (train_XX1[train_index], train_XX1[valid_index], 
                                        train_y[train_index], train_y[valid_index])
    filepath="weights_best.h5"
    sd = []    
    class LossHistory(Callback):
        def on_train_begin(self, logs={}):
            self.losses = [1,1]

        def on_epoch_end(self, batch, logs={}):
            self.losses.append(logs.get('loss'))
            sd.append(step_decay(len(self.losses)))
            print('lr:', step_decay(len(self.losses)))

    def step_decay(epoch):
        if (epoch != 0) & (epoch % 2 == 0):
            lrate = K.get_value(model.optimizer.lr)
            lrate = lrate * 0.5#/ (1 + decay_rate * epoch)            
            return lrate
        else:
            return K.get_value(model.optimizer.lr)
    l_rate=LearningRateScheduler(step_decay)
    class_weight = {0:.1,1:.9}
    history=LossHistory()
    model = lstmmodel()
    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=2, save_best_only=True, mode='min')
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=1, min_lr=0.0001, verbose=2)
    earlystopping = EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=2, verbose=2, mode='auto')
    callbacks = [checkpoint, reduce_lr]#, l_rate, history]
    np.random.seed(2018)
    if i == 0:print(model.summary()) 
    model.fit([X_train], Y_train, batch_size=512, epochs=5, validation_data=([X_val], Y_val), #class_weight=class_weight,
    verbose=2, callbacks=callbacks)
    model.load_weights(filepath)
    y_pred = model.predict([X_val], batch_size=1024, verbose=2)
    #y_test += np.squeeze(model.predict([test_XX1], batch_size=1024, verbose=1))/5
    f1, threshold = f1_smart(np.squeeze(Y_val), np.squeeze(y_pred))
    print('Optimal F1: {:.4f} at threshold: {:.4f}'.format(f1, threshold))
    y_test_whole[:,i] = (np.squeeze(model.predict([test_XX1], batch_size=1024, verbose=1))>=threshold).astype(int)
    bestscore.append(threshold)

#y_test = y_test.reshape((-1, 1))

y_test = np.mean(y_test_whole, axis = 1, keepdims = True)
pred_test_y = (y_test>=.2).astype(int)
sub['prediction'] = pred_test_y
sub.to_csv("submission.csv", index=False)                                        
