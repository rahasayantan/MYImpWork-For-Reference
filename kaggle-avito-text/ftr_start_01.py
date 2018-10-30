import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import re
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn import preprocessing, model_selection, metrics
import lightgbm as lgb
import gc

train_df = pd.read_csv('../input/train.csv',  parse_dates=["activation_date"])
test_df = pd.read_csv('../input/test.csv',  parse_dates=["activation_date"])

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np 
import seaborn as sns
import random 
import nltk
nltk.data.path.append("/media/sayantan/Personal/nltk_data")

from nltk.stem.snowball import RussianStemmer
from fuzzywuzzy import fuzz
from nltk.corpus import stopwords
from tqdm import tqdm
from scipy.stats import skew, kurtosis
from scipy.spatial.distance import cosine, cityblock, jaccard, canberra, euclidean, minkowski, braycurtis
from nltk import word_tokenize
stopwords = stopwords.words('russian')

def genFeatures(x):
    x["activation_weekday"] = x["activation_date"].dt.weekday
    x["monthday"] = x["activation_date"].dt.day
    x["weekinmonday"] = x["monthday"] // 7
##################Added in set 1 - 0.01 Improvement
    x['price_new'] = np.log1p(x.price) # log transform improves co-relation with deal_price
    x['count_null_in_row'] = x.isnull().sum(axis=1)# works
    x['has_description'] = x.description.isnull().astype(int) 
    x['has_image'] = x.image.isnull().astype(int) 
    x['has_image_top'] = x.image_top_1.isnull().astype(int) 
    x['has_param1'] = x.param_1.isnull().astype(int) 
    x['has_param2'] = x.param_2.isnull().astype(int) 
    x['has_param3'] = x.param_3.isnull().astype(int) 
    x['has_price'] = x.price.isnull().astype(int) 
#################Added in set 2 - 0.00x Improvement
    x["description"].fillna("NA", inplace=True)
    x["desc_nwords"] = x["description"].apply(lambda x: len(x.split()))
    x['len_description'] = x['description'].apply(lambda x: len(x))
    x["title_nwords"] = x["title"].apply(lambda x: len(x.split()))   
    x['len_title'] = x['title'].apply(lambda x: len(x))
    x['params'] = x['param_1'].fillna('') + ' ' + x['param_2'].fillna('') + ' ' + x['param_3'].fillna('')
    x['params'] = x['params'].str.strip()
    x['len_params'] = x['params'].apply(lambda x: len(x))
    x['words_params'] = x['params'].apply(lambda x: len(x.split()))
    x['symbol1_count'] = x['description'].str.count('↓')
    x['symbol2_count'] = x['description'].str.count('\*')
    x['symbol3_count'] = x['description'].str.count('✔')
    x['symbol4_count'] = x['description'].str.count('❀')
    x['symbol5_count'] = x['description'].str.count('➚')
    x['symbol6_count'] = x['description'].str.count('ஜ')
    x['symbol7_count'] = x['description'].str.count('.')
    x['symbol8_count'] = x['description'].str.count('!')
    x['symbol9_count'] = x['description'].str.count('\?')
    x['symbol10_count'] = x['description'].str.count('  ')
    x['symbol11_count'] = x['description'].str.count('-')
    x['symbol12_count'] = x['description'].str.count(',') 
####################    
    return x
    
train_df = genFeatures(train_df)
test_df = genFeatures(test_df)

test_df['deal_probability']=10.0

############################

english_stemmer = nltk.stem.SnowballStemmer('russian')

def clean_text(text):
    #text = re.sub(r'(\d+),(\d+)', r'\1.\2', text)
    text = text.replace(u'²', '2')
    text = text.lower()
    text = re.sub(u'[^a-zа-я0-9]', ' ', text)
    text = re.sub('\s+', ' ', text)
    return text.strip()

def stem_tokens(tokens, stemmer):
    stemmed = []
    for token in tokens:
        #stemmed.append(stemmer.lemmatize(token))
        stemmed.append(stemmer.stem(token))
    return stemmed

def preprocess_data(line,
                    exclude_stopword=True,
                    encode_digit=False):
    ## tokenize
    line = clean_text(line)
    tokens = [x.lower() for x in nltk.word_tokenize(line)]
    ## stem
    tokens_stemmed = stem_tokens(tokens, english_stemmer)#english_stemmer
    if exclude_stopword:
        tokens_stemmed = [x for x in tokens_stemmed if x not in stopwords]
    return ' '.join(tokens_stemmed)

train_test = pd.concat((train_df, test_df), axis = 'rows')

## After cleaning => then find intersection
train_test["title_clean"]= list(train_test[["title"]].apply(lambda x: preprocess_data(x["title"]), axis=1))
train_test["desc_clean"]= list(train_test[["description"]].apply(lambda x: preprocess_data(x["description"]), axis=1))
train_test["params_clean"]= list(train_test[["params"]].apply(lambda x: preprocess_data(x["params"]), axis=1))

train_test['count_common_words_title_desc'] = train_test.apply(lambda x: len(set(str(x['title_clean']).lower().split()).intersection(set(str(x['desc_clean']).lower().split()))), axis=1)
train_test['count_common_words_title_params'] = train_test.apply(lambda x: len(set(str(x['title_clean']).lower().split()).intersection(set(str(x['params_clean']).lower().split()))), axis=1)
train_test['count_common_words_params_desc'] = train_test.apply(lambda x: len(set(str(x['params_clean']).lower().split()).intersection(set(str(x['desc_clean']).lower().split()))), axis=1)

print("Cleaned texts..")
###################

# Count Nouns
import pymorphy2
morph = pymorphy2.MorphAnalyzer(result_type=None)
from fastcache import clru_cache as lru_cache

@lru_cache(maxsize=1000000)
def lemmatize_pos(word):
    _, tag, norm_form, _, _ = morph.parse(word)[0]
    return norm_form, tag.POS

def getPOS(x, pos1 = 'NOUN'):
    lemmatized = []
    x = clean_text(x)
    #x = re.sub(u'[.]', ' ', x)
    for s in x.split():
        s, pos = lemmatize_pos(s)
        if pos != None:
            if pos1 in pos:
                lemmatized.append(s)
    return ' '.join(lemmatized)

train_test['get_nouns_title'] = list(train_test.apply(lambda x: getPOS(x['title'], 'NOUN'), axis=1))
train_test['get_nouns_desc'] = list(train_test.apply(lambda x: getPOS(x['description'], 'NOUN'), axis=1))

train_test['get_adj_title'] = list(train_test.apply(lambda x: getPOS(x['title'], 'ADJ'), axis=1))
train_test['get_adj_desc'] = list(train_test.apply(lambda x: getPOS(x['description'], 'ADJ'), axis=1))

train_test['get_verb_title'] = list(train_test.apply(lambda x: getPOS(x['title'], 'VERB'), axis=1))
train_test['get_verb_desc'] = list(train_test.apply(lambda x: getPOS(x['description'], 'VERB'), axis=1))

# Count digits
def count_digit(x):
    x = clean_text(x)    
    return len(re.findall(r'\b\d+\b', x))
    
train_test['count_of_digit_in_title'] = list(train_test.apply(lambda x: count_digit(x['title']), axis=1))
train_test['count_of_digit_in_desc'] = list(train_test.apply(lambda x: count_digit(x['description']), axis=1))
train_test['count_of_digit_in_params'] = list(train_test.apply(lambda x: count_digit(x['params']), axis=1))

## get unicode features
count_unicode = lambda x: len([c for c in x if ord(c) > 1105])
count_distunicode = lambda x: len({c for c in x if ord(c) > 1105})

train_test['count_of_unicode_in_title'] = list(train_test.apply(lambda x: count_unicode(x['title']), axis=1))
train_test['count_of_unicode_in_desc'] = list(train_test.apply(lambda x: count_distunicode(x['description']), axis=1))
train_test['count_of_distuni_in_title'] = list(train_test.apply(lambda x: count_unicode(x['title']), axis=1))
train_test['count_of_distuni_in_desc'] = list(train_test.apply(lambda x: count_distunicode(x['description']), axis=1))

###
count_caps = lambda x: len([c for c in x if c.isupper()])
train_test['count_caps_in_title'] = list(train_test.apply(lambda x: count_caps(x['title']), axis=1))
train_test['count_caps_in_desc'] = list(train_test.apply(lambda x: count_caps(x['description']), axis=1))

import string
count_punct = lambda x: len([c for c in x if c in string.punctuation])
train_test['count_punct_in_title'] = list(train_test.apply(lambda x: count_punct(x['title']), axis=1))
train_test['count_punct_in_desc'] = list(train_test.apply(lambda x: count_punct(x['description']), axis=1))

print("Computed POS Features and others..")

train_test['count_common_nouns'] = train_test.apply(lambda x: len(set(str(x['get_nouns_title']).lower().split()).intersection(set(str(x['get_nouns_desc']).lower().split()))), axis=1)
train_test['count_common_adj'] = train_test.apply(lambda x: len(set(str(x['get_adj_title']).lower().split()).intersection(set(str(x['get_adj_desc']).lower().split()))), axis=1)

train_test['ratio_of_unicode_in_title'] = train_test['count_of_unicode_in_title'] / train_test['len_title']
train_test['ratio_of_unicode_in_desc'] = train_test['count_of_unicode_in_desc'] / train_test['len_description']

train_test['ratio_of_punct_in_title'] = train_test['count_punct_in_title'] / train_test['len_title']
train_test['ratio_of_punct_in_desc'] = train_test['count_punct_in_desc'] / train_test['len_description']

train_test['ratio_of_cap_in_title'] = train_test['count_caps_in_title'] / train_test['len_title']
train_test['ratio_of_cap_in_desc'] = train_test['count_caps_in_desc'] / train_test['len_description']

train_test['count_nouns_in_title'] = train_test["get_nouns_title"].apply(lambda x: len(x.split()))
train_test['count_nouns_in_desc'] = train_test['get_nouns_desc'].apply(lambda x: len(x.split()))

train_test['count_adj_in_title'] = train_test["get_adj_title"].apply(lambda x: len(x.split()))
train_test['count_adj_in_desc'] = train_test['get_adj_desc'].apply(lambda x: len(x.split()))

train_test['count_verb_title'] = train_test['get_verb_title'].apply(lambda x: len(x.split()))
train_test['count_verb_desc'] = train_test['get_verb_desc'].apply(lambda x: len(x.split()))

train_test['ratio_nouns_in_title'] = train_test["count_nouns_in_title"] / train_test["title_nwords"]
train_test['ratio_nouns_in_desc'] = train_test["count_nouns_in_desc"] / train_test["desc_nwords"]

train_test['ratio_adj_in_title'] = train_test["count_adj_in_title"] / train_test["title_nwords"]
train_test['ratio_adj_in_desc'] = train_test["count_adj_in_desc"] / train_test["desc_nwords"]

train_test['ratio_vrb_in_title'] = train_test["count_verb_title"] / train_test["title_nwords"]
train_test['ratio_vrb_in_desc'] = train_test["count_verb_desc"] / train_test["desc_nwords"]

train_test["title"]= list(train_test[["title"]].apply(lambda x: clean_text(x["title"]), axis=1))
train_test["description"]= list(train_test[["description"]].apply(lambda x: clean_text(x["description"]), axis=1))
train_test["params"]= list(train_test[["params"]].apply(lambda x: clean_text(x["params"]), axis=1))
#######################
### Save
#######################
train_df = train_test.loc[train_test.deal_probability != 10].reset_index(drop = True)
test_df = train_test.loc[train_test.deal_probability == 10].reset_index(drop = True)

for c in train_df.columns:
    if train_df[c].dtype == 'float64':
        train_df[c] = train_df[c].astype('float32')
        test_df[c] = test_df[c].astype('float32')

train_df.to_feather('../train_basic_features.pkl')
test_df.to_feather('../test__basic_features.pkl')


#######################
### Label Enc
#######################
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder, MinMaxScaler

cat_vars = ["user_id", "region", "city", "parent_category_name", "category_name", "user_type", "param_1", "param_2", "param_3"]
for col in cat_vars:
    lbl = preprocessing.LabelEncoder()
    lbl.fit(list(train_df[col].values.astype('str')) + list(test_df[col].values.astype('str')))
    train_df[col] = lbl.transform(list(train_df[col].values.astype('str')))
    test_df[col] = lbl.transform(list(test_df[col].values.astype('str')))

train_df.to_feather('../train_basic_features_lblencCats.pkl')
test_df.to_feather('../test__basic_features_lblencCats.pkl')


#######################
### One hots
#######################
train_df=pd.read_feather('../train_basic_features_lblencCats.pkl')
test_df=pd.read_feather('../test__basic_features_lblencCats.pkl')

from sklearn.externals import joblib
le = OneHotEncoder()
X = le.fit_transform(np.array(train_df.user_id.values.tolist() + test_df.user_id.values.tolist()).reshape(-1,1))
joblib.dump(X, "../user_id_onehot.pkl")

X = le.fit_transform(np.array(train_df.region.values.tolist() + test_df.region.values.tolist()).reshape(-1,1))
joblib.dump(X, "../region_onehot.pkl")

X = le.fit_transform(np.array(train_df.city.values.tolist() + test_df.city.values.tolist()).reshape(-1,1))
joblib.dump(X, "../city_onehot.pkl")

X = le.fit_transform(np.array(train_df.parent_category_name.values.tolist() + test_df.parent_category_name.values.tolist()).reshape(-1,1))
joblib.dump(X, "../parent_category_name_onehot.pkl")

X = le.fit_transform(np.array(train_df.category_name.values.tolist() + test_df.category_name.values.tolist()).reshape(-1,1))
joblib.dump(X, "../category_name_onehot.pkl")

X = le.fit_transform(np.array(train_df.user_type.values.tolist() + test_df.user_type.values.tolist()).reshape(-1,1))
joblib.dump(X, "../user_type_onehot.pkl")

X = le.fit_transform(np.array(train_df.param_1.values.tolist() + test_df.param_1.values.tolist()).reshape(-1,1))
joblib.dump(X, "../param_1_onehot.pkl")

X = le.fit_transform(np.array(train_df.param_2.values.tolist() + test_df.param_2.values.tolist()).reshape(-1,1))
joblib.dump(X, "../param_2_onehot.pkl")

X = le.fit_transform(np.array(train_df.param_3.values.tolist() + test_df.param_3.values.tolist()).reshape(-1,1))
joblib.dump(X, "../param_3_onehot.pkl")

train_df.drop(cat_vars, inplace = True, axis = 'columns')
test_df.drop(cat_vars, inplace = True, axis = 'columns')

train_df.to_feather('../train_basic_features_woCats.pkl')
test_df.to_feather('../test__basic_features_woCats.pkl')

#######################
### Tfidf
#######################
train_df=pd.read_feather('../train_basic_features_woCats.pkl')
test_df=pd.read_feather('../test__basic_features_woCats.pkl')
from sklearn.externals import joblib
### TFIDF Vectorizer ###
train_df['params'] = train_df['params'].fillna('NA')
test_df['params'] = test_df['params'].fillna('NA')

tfidf_vec = TfidfVectorizer(ngram_range=(1,3),max_features = 10000,#min_df=3, max_df=.85,
                                 analyzer='word', token_pattern= r'\w{1,}',
                                 use_idf=1, smooth_idf=0, sublinear_tf=1,)
                                 #TfidfVectorizer(ngram_range=(1,2))
full_tfidf = tfidf_vec.fit_transform(train_df['params'].values.tolist() + test_df['params'].values.tolist())
train_tfidf = tfidf_vec.transform(train_df['params'].values.tolist())
test_tfidf = tfidf_vec.transform(test_df['params'].values.tolist())

del full_tfidf
print("TDIDF Params UNCLEAN..")
joblib.dump([train_tfidf, test_tfidf], "../params_tfidf.pkl")

### TFIDF Vectorizer ###
train_df['title_clean'] = train_df['title_clean'].fillna('NA')
test_df['title_clean'] = test_df['title_clean'].fillna('NA')

tfidf_vec = TfidfVectorizer(ngram_range=(1,2),max_features = 20000,#,min_df=3, max_df=.85,
                                 analyzer='word', token_pattern= r'\w{1,}',
                                 use_idf=1, smooth_idf=0, sublinear_tf=1,)
full_tfidf = tfidf_vec.fit_transform(train_df['title_clean'].values.tolist() + test_df['title_clean'].values.tolist())
train_tfidf = tfidf_vec.transform(train_df['title_clean'].values.tolist())
test_tfidf = tfidf_vec.transform(test_df['title_clean'].values.tolist())
joblib.dump([train_tfidf, test_tfidf], "../title_tfidf.pkl")

del full_tfidf

print("TDIDF TITLE CLEAN..")

### TFIDF Vectorizer ###
train_df['desc_clean'] = train_df['desc_clean'].fillna(' ')
test_df['desc_clean'] = test_df['desc_clean'].fillna(' ')

tfidf_vec = TfidfVectorizer(ngram_range=(1,2), max_features = 20000, #,min_df=3, max_df=.85,
                                 analyzer='word', token_pattern= r'\w{1,}',
                                 use_idf=1, smooth_idf=0, sublinear_tf=1,)
full_tfidf = tfidf_vec.fit_transform(train_df['desc_clean'].values.tolist() + test_df['desc_clean'].values.tolist())
train_tfidf = tfidf_vec.transform(train_df['desc_clean'].values.tolist())
test_tfidf = tfidf_vec.transform(test_df['desc_clean'].values.tolist())
joblib.dump([train_tfidf, test_tfidf], "../desc_tfidf.pkl")
del full_tfidf
print("TDIDF DESC CLEAN..")

### TFIDF Vectorizer ###
train_df['get_nouns_title'] = train_df['get_nouns_title'].fillna(' ')
test_df['get_nouns_title'] = test_df['get_nouns_title'].fillna(' ')

tfidf_vec = TfidfVectorizer(ngram_range=(1,1), max_features = 10000)
full_tfidf = tfidf_vec.fit_transform(train_df['get_nouns_title'].values.tolist() + test_df['get_nouns_title'].values.tolist())
train_tfidf = tfidf_vec.transform(train_df['get_nouns_title'].values.tolist())
test_tfidf = tfidf_vec.transform(test_df['get_nouns_title'].values.tolist())
joblib.dump([train_tfidf, test_tfidf], "../nouns_title_tfidf.pkl")
del full_tfidf
print("TDIDF Title Noun..")

### TFIDF Vectorizer ###
train_df['get_nouns_desc'] = train_df['get_nouns_desc'].fillna(' ')
test_df['get_nouns_desc'] = test_df['get_nouns_desc'].fillna(' ')

tfidf_vec = TfidfVectorizer(ngram_range=(1,1), max_features = 10000)
full_tfidf = tfidf_vec.fit_transform(train_df['get_nouns_desc'].values.tolist() + test_df['get_nouns_desc'].values.tolist())
train_tfidf = tfidf_vec.transform(train_df['get_nouns_desc'].values.tolist())
test_tfidf = tfidf_vec.transform(test_df['get_nouns_desc'].values.tolist())
joblib.dump([train_tfidf, test_tfidf], "../nouns_desc_tfidf.pkl")
del full_tfidf
print("TDIDF Desc Noun..")

### TFIDF Vectorizer ###
train_df['get_adj_title'] = train_df['get_adj_title'].fillna(' ')
test_df['get_adj_title'] = test_df['get_adj_title'].fillna(' ')

tfidf_vec = TfidfVectorizer(ngram_range=(1,1), max_features = 10000)
full_tfidf = tfidf_vec.fit_transform(train_df['get_adj_title'].values.tolist() + test_df['get_adj_title'].values.tolist())
train_tfidf = tfidf_vec.transform(train_df['get_adj_title'].values.tolist())
test_tfidf = tfidf_vec.transform(test_df['get_adj_title'].values.tolist())
joblib.dump([train_tfidf, test_tfidf], "../adj_title_tfidf.pkl")
del full_tfidf
print("TDIDF TITLE Adj..")

### TFIDF Vectorizer ###
train_df['get_adj_desc'] = train_df['get_adj_desc'].fillna(' ')
test_df['get_adj_desc'] = test_df['get_adj_desc'].fillna(' ')

tfidf_vec = TfidfVectorizer(ngram_range=(1,1), max_features = 10000)
full_tfidf = tfidf_vec.fit_transform(train_df['get_adj_desc'].values.tolist() + test_df['get_adj_desc'].values.tolist())
train_tfidf = tfidf_vec.transform(train_df['get_adj_desc'].values.tolist())
test_tfidf = tfidf_vec.transform(test_df['get_adj_desc'].values.tolist())
joblib.dump([train_tfidf, test_tfidf], "../adj_desc_tfidf.pkl")
del full_tfidf
print("TDIDF Desc Adj..")


### TFIDF Vectorizer ###
train_df['get_verb_title'] = train_df['get_verb_title'].fillna(' ')
test_df['get_verb_title'] = test_df['get_verb_title'].fillna(' ')

tfidf_vec = TfidfVectorizer(ngram_range=(1,1), max_features = 10000)
full_tfidf = tfidf_vec.fit_transform(train_df['get_verb_title'].values.tolist() + test_df['get_verb_title'].values.tolist())
train_tfidf = tfidf_vec.transform(train_df['get_verb_title'].values.tolist())
test_tfidf = tfidf_vec.transform(test_df['get_verb_title'].values.tolist())
joblib.dump([train_tfidf, test_tfidf], "../verb_title_tfidf.pkl")
del full_tfidf
print("TDIDF TITLE Verb..")

### TFIDF Vectorizer ###
train_df['get_verb_desc'] = train_df['get_verb_desc'].fillna(' ')
test_df['get_verb_desc'] = test_df['get_verb_desc'].fillna(' ')

tfidf_vec = TfidfVectorizer(ngram_range=(1,1), max_features = 10000)
full_tfidf = tfidf_vec.fit_transform(train_df['get_verb_desc'].values.tolist() + test_df['get_verb_desc'].values.tolist())
train_tfidf = tfidf_vec.transform(train_df['get_verb_desc'].values.tolist())
test_tfidf = tfidf_vec.transform(test_df['get_verb_desc'].values.tolist())
joblib.dump([train_tfidf, test_tfidf], "../verb_desc_tfidf.pkl")
del full_tfidf
print("TDIDF Desc Verb..")

###############################
# Sentence to seq
###############################
print('Generate Word Sequences')
train_df=pd.read_feather('../train_basic_features_woCats.pkl')
test_df=pd.read_feather('../test__basic_features_woCats.pkl')

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
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

#######OHC WeekDay
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder, MinMaxScaler

le = OneHotEncoder()
X = le.fit_transform(np.array(train_df.activation_weekday.values.tolist() + test_df.activation_weekday.values.tolist()).reshape(-1,1))


################################################
# Cat encoding
################################################
train_df=pd.read_feather('../train_basic_features.pkl')
test_df=pd.read_feather('../test__basic_features.pkl')

def catEncode(train_char, test_char, y, colLst = [], nbag = 10, nfold = 20, minCount = 3, postfix = ''):
    train_df = train_char.copy()
    test_df = test_char.copy()
    if not colLst:
        print("Empty ColLst")
        for c in train_char.columns:
            data = train_char[[c]].copy()
            data['y'] = y
            enc_mat = np.zeros((y.shape[0],4))
            enc_mat_test = np.zeros((test_char.shape[0],4))
            for bag in np.arange(nbag):
                kf = model_selection.KFold(n_splits= nfold, shuffle=True, random_state=2017*bag)
                for dev_index, val_index in kf.split(range(data['y'].shape[0])):
                    dev_X, val_X = data.iloc[dev_index,:], data.iloc[val_index,:]
                    datax = dev_X.groupby([c]).agg([len,np.mean,np.std, np.median])
                    datax.columns = ['_'.join(col).strip() for col in datax.columns.values]
#                    datax = datax.loc[datax.y_len > minCount]
                    ind = c + postfix
                    datax.rename(columns = {'y_mean': ('y_mean_' + ind), 'y_std': ('y_std_' + ind),
                    'y_len_': ('y_len' + ind), 'y_median_': ('y_median' + ind),}, inplace = True)
#                    datax[c+'_medshftenc'] =  datax['y_median']-med_y
#                    datax.drop(['y_len','y_mean','y_std','y_median'],axis=1,inplace=True)
                    datatst = test_char[[c]].copy()
                    val_X = val_X.join(datax,on=[c], how='left').fillna(np.mean(y))
                    datatst = datatst.join(datax,on=[c], how='left').fillna(np.mean(y))
                    enc_mat[val_index,...] += val_X[list(set(datax.columns)-set([c]))]
                    enc_mat_test += datatst[list(set(datax.columns)-set([c]))]
            enc_mat_test /= (nfold * nbag)
            enc_mat /= (nbag)        
            enc_mat = pd.DataFrame(enc_mat)  
            enc_mat.columns=[ind + str(x) for x in list(set(datax.columns)-set([c]))] 
            enc_mat_test = pd.DataFrame(enc_mat_test)  
            enc_mat_test.columns=enc_mat.columns
            train_df = pd.concat((enc_mat.reset_index(drop = True),train_df.reset_index(drop = True)), axis=1)
            test_df = pd.concat([enc_mat_test.reset_index(drop = True),test_df.reset_index(drop = True)],axis=1)
        
    else:
        print("Not Empty ColLst")
        data = train_char[colLst].copy()
        data['y'] = y
        enc_mat = np.zeros((y.shape[0],4))
        enc_mat_test = np.zeros((test_char.shape[0],4))
        for bag in np.arange(nbag):     
            kf = model_selection.KFold(n_splits= nfold, shuffle=True, random_state=2017*bag)
            for dev_index, val_index in kf.split(range(data['y'].shape[0])):
                dev_X, val_X = data.iloc[dev_index,:], data.iloc[val_index,:]
                datax = dev_X.groupby(colLst).agg([len,np.mean,np.std, np.median])
                datax.columns = ['_'.join(col).strip() for col in datax.columns.values]
#                datax = datax.loc[datax.y_len > minCount]
                ind = '_'.join(colLst) + postfix
                datax.rename(columns = {'y_mean': ('y_mean_' + ind), 'y_std': ('y_std_' + ind),
                'y_len': ('y_len_' + ind), 'y_median': ('y_median_' + ind),}, inplace = True)
                datatst = test_char[colLst].copy()
                val_X = val_X.join(datax,on=colLst, how='left').fillna(np.mean(y))
                datatst = datatst.join(datax,on=colLst, how='left').fillna(np.mean(y))
                print(val_X[list(set(datax.columns)-set(colLst))].columns)
                enc_mat[val_index,...] += val_X[list(set(datax.columns)-set(colLst))]
                enc_mat_test += datatst[list(set(datax.columns)-set(colLst))]
                
        enc_mat_test /= (nfold * nbag)
        enc_mat /= (nbag)        
        enc_mat = pd.DataFrame(enc_mat)  
        enc_mat.columns=[ind + str(x) for x in list(set(datax.columns)-set([c]))] 
        enc_mat_test = pd.DataFrame(enc_mat_test)  
        enc_mat_test.columns=enc_mat.columns 
        train_df = pd.concat((enc_mat.reset_index(drop = True),train_df.reset_index(drop = True)), axis=1)
        test_df = pd.concat([enc_mat_test.reset_index(drop = True),test_df.reset_index(drop = True)],axis=1)
    print(train_df.columns)
    print(test_df.columns)                            
    for c in train_df.columns:
        if train_df[c].dtype == 'float64':
            train_df[c] = train_df[c].astype('float32')
            test_df[c] = test_df[c].astype('float32')
    return train_df, test_df

catCols = ['user_id', 'region', 'city', 'parent_category_name',
       'category_name', 'user_type']

train_df, test_df = catEncode(train_df[catCols].copy(), test_df[catCols].copy(), train_df.deal_probability.values, nbag = 10, nfold = 10, minCount = 0)

train_df.to_feather('../train_cat_targetenc.pkl')
test_df.to_feather('../test_cat_targetenc.pkl')

################################################################
# Tfidf - part 2
################################################################

import os; os.environ['OMP_NUM_THREADS'] = '1'
from sklearn.decomposition import TruncatedSVD
import nltk
nltk.data.path.append("/media/sayantan/Personal/nltk_data")
from nltk.stem.snowball import RussianStemmer
from nltk.corpus import stopwords
import time
from typing import List, Dict
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer as Tfidf
from sklearn.model_selection import KFold
from sklearn.externals import joblib
from scipy.sparse import hstack, csr_matrix
import pandas as pd
import numpy as np
import gc
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder, MinMaxScaler
from sklearn import model_selection

english_stemmer = nltk.stem.SnowballStemmer('russian')

def clean_text(text):
    #text = re.sub(r'(\d+),(\d+)', r'\1.\2', text)
    text = text.replace(u'²', '2')
    text = text.lower()
    text = re.sub(u'[^a-zа-я0-9]', ' ', text)
    text = re.sub('\s+', ' ', text)
    return text.strip()

def stem_tokens(tokens, stemmer):
    stemmed = []
    for token in tokens:
        #stemmed.append(stemmer.lemmatize(token))
        stemmed.append(stemmer.stem(token))
    return stemmed

def preprocess_data(line,
                    exclude_stopword=True,
                    encode_digit=False):
    ## tokenize
    line = clean_text(line)
    tokens = [x.lower() for x in nltk.word_tokenize(line)]
    ## stem
    tokens_stemmed = stem_tokens(tokens, english_stemmer)#english_stemmer
    if exclude_stopword:
        tokens_stemmed = [x for x in tokens_stemmed if x not in stopwords]
    return ' '.join(tokens_stemmed)

stopwords = stopwords.words('russian')
train_per=pd.read_csv('../input/train_active.csv', usecols = ['param_1', 'param_2', 'param_3'])#,'title','description'])
test_per=pd.read_csv('../input/test_active.csv', usecols = ['param_1', 'param_2', 'param_3'])#,'title','description'])
train_test = pd.concat((train_per, test_per), axis = 'rows')
del train_per, test_per; gc.collect()

train_test['params'] = train_test['param_1'].fillna('') + ' ' + train_test['param_2'].fillna('') + ' ' + train_test['param_3'].fillna('')
import re
train_test.drop(['param_1', 'param_2', 'param_3'], axis = 'columns', inplace=True)
train_test["params"]= list(train_test[["params"]].apply(lambda x: clean_text(x["params"]), axis=1))
import re
train_df=pd.read_feather('../train_basic_features_woCats.pkl')
test_df=pd.read_feather('../test__basic_features_woCats.pkl')
from sklearn.externals import joblib

### TFIDF Vectorizer ###
train_df['params'] = train_df['params'].fillna('NA')
test_df['params'] = test_df['params'].fillna('NA')

tfidf_vec = TfidfVectorizer(ngram_range=(1,3),max_features = 10000,#min_df=3, max_df=.85,
                                 analyzer='word', token_pattern= r'\w{1,}',
                                 use_idf=1, smooth_idf=0, sublinear_tf=1,)
                                 #TfidfVectorizer(ngram_range=(1,2))
full_tfidf = tfidf_vec.fit_transform(train_test['params'].values.tolist() + train_df['params'].values.tolist() + test_df['params'].values.tolist())
train_tfidf = tfidf_vec.transform(train_df['params'].values.tolist())
test_tfidf = tfidf_vec.transform(test_df['params'].values.tolist())

del full_tfidf
print("TDIDF Params UNCLEAN..")
joblib.dump([train_tfidf, test_tfidf], "../params_tfidf2.pkl")

tfidf_vec = TfidfVectorizer(ngram_range=(1,1),max_features = 10000,max_df=.4,#min_df=3, 
                                 analyzer='word', token_pattern= r'\w{1,}',
                                 use_idf=1, smooth_idf=0, sublinear_tf=1,)
                                 #TfidfVectorizer(ngram_range=(1,2))
full_tfidf = tfidf_vec.fit_transform(train_test['params'].values.tolist() + train_df['params'].values.tolist() + test_df['params'].values.tolist())
train_tfidf = tfidf_vec.transform(train_df['params'].values.tolist())
test_tfidf = tfidf_vec.transform(test_df['params'].values.tolist())

del full_tfidf
print("TDIDF Params UNCLEAN..")
joblib.dump([train_tfidf, test_tfidf], "../params_tfidf3.pkl")

del(train_test); gc.collect()

train_per=pd.read_csv('../input/train_active.csv', usecols = ['title'])#,'title','description'])
test_per=pd.read_csv('../input/test_active.csv', usecols = ['title'])#,'title','description'])
train_test = pd.concat((train_per, test_per), axis = 'rows')
del train_per, test_per; gc.collect()

train_test.fillna('NA', inplace=True)
train_test["title_clean"]= list(train_test[["title"]].apply(lambda x: preprocess_data(x["title"]), axis=1))
train_df['title_clean'] = train_df['title_clean'].fillna('NA')
test_df['title_clean'] = test_df['title_clean'].fillna('NA')

tfidf_vec = TfidfVectorizer(ngram_range=(1,2),max_features = 20000,#,min_df=3, max_df=.85,
                                 analyzer='word', token_pattern= r'\w{1,}',
                                 use_idf=1, smooth_idf=0, sublinear_tf=1,)
full_tfidf = tfidf_vec.fit_transform(train_test['title_clean'].values.tolist()+train_df['title_clean'].values.tolist() + test_df['title_clean'].values.tolist())
train_tfidf = tfidf_vec.transform(train_df['title_clean'].values.tolist())
test_tfidf = tfidf_vec.transform(test_df['title_clean'].values.tolist())
joblib.dump([train_tfidf, test_tfidf], "../title_tfidf2.pkl")

del full_tfidf

print("TDIDF TITLE CLEAN..")
train_df['title_clean'] = train_df['title_clean'].fillna('NA')
test_df['title_clean'] = test_df['title_clean'].fillna('NA')

tfidf_vec = TfidfVectorizer(ngram_range=(1,1),max_features = 20000, max_df=.4,#,min_df=3,
                                 analyzer='word', token_pattern= r'\w{1,}',
                                 use_idf=1, smooth_idf=0, sublinear_tf=1,)
full_tfidf = tfidf_vec.fit_transform(train_test['title_clean'].values.tolist()+train_df['title_clean'].values.tolist() + test_df['title_clean'].values.tolist())
train_tfidf = tfidf_vec.transform(train_df['title_clean'].values.tolist())
test_tfidf = tfidf_vec.transform(test_df['title_clean'].values.tolist())
joblib.dump([train_tfidf, test_tfidf], "../title_tfidf3.pkl")

del full_tfidf

print("TDIDF TITLE CLEAN..")


del(train_test); gc.collect()

###Too slow###
'''
train_per=pd.read_csv('../input/train_active.csv', usecols = ['description'])#,'title','description'])
test_per=pd.read_csv('../input/test_active.csv', usecols = ['description'])#,'title','description'])
train_per.fillna(' ', inplace=True)
test_per.fillna(' ', inplace=True)

train_test["desc_clean"]= list(train_test[["description"]].apply(lambda x: preprocess_data(x["description"]), axis=1))

### TFIDF Vectorizer ###
train_df['desc_clean'] = train_df['desc_clean'].fillna(' ')
test_df['desc_clean'] = test_df['desc_clean'].fillna(' ')

tfidf_vec = TfidfVectorizer(ngram_range=(1,2), max_features = 20000, stop_words = stopwords#,min_df=3, 
                                 analyzer='word', token_pattern= r'\w{1,}',
                                 use_idf=1, smooth_idf=0, sublinear_tf=1,)
full_tfidf = tfidf_vec.fit_transform(train_test['desc_clean'].values.tolist()+train_df['desc_clean'].values.tolist() + test_df['desc_clean'].values.tolist())
train_tfidf = tfidf_vec.transform(train_df['desc_clean'].values.tolist())
test_tfidf = tfidf_vec.transform(test_df['desc_clean'].values.tolist())
joblib.dump([train_tfidf, test_tfidf], "../desc_tfidf2.pkl")
del full_tfidf
print("TDIDF DESC CLEAN..")
tfidf_vec = TfidfVectorizer(ngram_range=(1,1), max_features = 20000, max_df=.4,#,min_df=3, 
                                 analyzer='word', token_pattern= r'\w{1,}',
                                 use_idf=1, smooth_idf=0, sublinear_tf=1,)
full_tfidf = tfidf_vec.fit_transform(train_test['desc_clean'].values.tolist()+train_df['desc_clean'].values.tolist() + test_df['desc_clean'].values.tolist())
train_tfidf = tfidf_vec.transform(train_df['desc_clean'].values.tolist())
test_tfidf = tfidf_vec.transform(test_df['desc_clean'].values.tolist())
joblib.dump([train_tfidf, test_tfidf], "../desc_tfidf3.pkl")
del full_tfidf
print("TDIDF DESC CLEAN..")
'''
##########################################
# 13. Chargram -- too slow
##########################################
from collections import Counter
train_df=pd.read_feather('../train_basic_features_woCats.pkl')
test_df=pd.read_feather('../test__basic_features_woCats.pkl')

def char_ngrams(s):
    s = s.lower()
    s = s.replace(u' ', '')
    result = Counter()
    len_s = len(s)
    for n in [3, 4, 5]:
        result.update(s[i:i+n] for i in range(len_s - n + 1))
    return ' '.join(list(result))

data = pd.concat((train_df, test_df), axis = 'rows')

data['param_chargram'] = list(data[['params']].apply(lambda x: char_ngrams(x['params']), axis=1))
data['title_chargram'] = list(data[['title']].apply(lambda x: char_ngrams(x['title']), axis=1))
#data['desc_chargram'] = list(data[['description']].apply(lambda x: char_ngrams(x['description']), axis=1))

#data['count_common_chargram'] = data.apply(lambda x: len(set(str(x['title_chargram']).lower().split()).intersection(set(str(x['desc_chargram']).lower().split()))), axis=1)

train_df = data.loc[data.deal_probability != 10].reset_index(drop = True)
test_df = data.loc[data.deal_probability == 10].reset_index(drop = True)

del(data); gc.collect()

#####Chargram -TFIDF
tfidf_vec = TfidfVectorizer(ngram_range=(1,3),max_features = 10000, min_df=3, max_df=.75)
full_tfidf = tfidf_vec.fit_transform(train_df['title_chargram'].values.tolist() + test_df['title_chargram'].values.tolist())
train_tfidf = tfidf_vec.transform(train_df['title_chargram'].values.tolist())
test_tfidf = tfidf_vec.transform(test_df['title_chargram'].values.tolist())
from sklearn.externals import joblib 
joblib.dump([train_tfidf, test_tfidf], '../title_chargram_tfidf.pkl')

tfidf_vec = TfidfVectorizer(ngram_range=(1,3),max_features = 10000, min_df=3, max_df=.75)
full_tfidf = tfidf_vec.fit_transform(train_df['param_chargram'].values.tolist() + test_df['param_chargram'].values.tolist())
train_tfidf = tfidf_vec.transform(train_df['param_chargram'].values.tolist())
test_tfidf = tfidf_vec.transform(test_df['param_chargram'].values.tolist())
from sklearn.externals import joblib 
joblib.dump([train_tfidf, test_tfidf], '../param_chargram_tfidf.pkl')


#######Chargram of Cat and Parent cat
def clean_text(text):
    #text = re.sub(r'(\d+),(\d+)', r'\1.\2', text)
    text = text.replace(u'²', '2')
    text = text.lower()
    text = re.sub(u'[^a-zа-я0-9]', ' ', text)
    text = re.sub('\s+', ' ', text)
    return text.strip()

train_df = pd.read_feather('../train_basic_features.pkl')
test_df = pd.read_feather('../test__basic_features.pkl')
data = pd.concat([train_df, test_df], axis= 'rows')
data['categories'] = data["parent_category_name"].fillna(' ') + data["category_name"].fillna(' ') 
data['cat_chargram'] = list(data[['categories']].apply(lambda x: char_ngrams(x['categories']), axis=1))
train_df = data.loc[data.deal_probability != 10].reset_index(drop = True)
test_df = data.loc[data.deal_probability == 10].reset_index(drop = True)

del(data); gc.collect()
tfidf_vec = TfidfVectorizer(ngram_range=(1,3),max_features = 1000, min_df=3, max_df=.75)
full_tfidf = tfidf_vec.fit_transform(train_df['cat_chargram'].values.tolist() + test_df['cat_chargram'].values.tolist())
train_tfidf = tfidf_vec.transform(train_df['cat_chargram'].values.tolist())
test_tfidf = tfidf_vec.transform(test_df['cat_chargram'].values.tolist())
from sklearn.externals import joblib 
joblib.dump([train_tfidf, test_tfidf], '../cat_chargram_tfidf.pkl')


##############################
## New Kaggle Ftr
##############################
import pandas as pd
import gc
used_cols = ['item_id', 'user_id']

train = pd.read_csv('../input/train.csv', usecols=used_cols)
train_active = pd.read_csv('../input/train_active.csv', usecols=used_cols)
test = pd.read_csv('../input/test.csv', usecols=used_cols)
test_active = pd.read_csv('../input/test_active.csv', usecols=used_cols)

train_periods = pd.read_csv('../input/periods_train.csv', parse_dates=['date_from', 'date_to'])
test_periods = pd.read_csv('../input/periods_test.csv', parse_dates=['date_from', 'date_to'])

train.head()

all_samples = pd.concat([
    train,
    train_active,
    test,
    test_active
]).reset_index(drop=True)
all_samples.drop_duplicates(['item_id'], inplace=True)

del train_active
del test_active
gc.collect()

all_periods = pd.concat([
    train_periods,
    test_periods
])

del train_periods
del test_periods
gc.collect()

all_periods.head()

all_periods['days_up'] = (all_periods['date_to'] - all_periods['date_from']).dt.days

gp = all_periods.groupby(['item_id'])[['days_up']]

gp_df = pd.DataFrame()
gp_df['days_up_sum'] = gp.sum()['days_up']
gp_df['times_put_up'] = gp.count()['days_up']
gp_df.reset_index(inplace=True)
gp_df.rename(index=str, columns={'index': 'item_id'})

gp_df.head()

all_periods.drop_duplicates(['item_id'], inplace=True)
all_periods = all_periods.merge(gp_df, on='item_id', how='left')
all_periods.head()

del gp
del gp_df
gc.collect()

all_periods = all_periods.merge(all_samples, on='item_id', how='left')
all_periods.head()

gp = all_periods.groupby(['user_id'])[['days_up_sum', 'times_put_up']].mean().reset_index() \
    .rename(index=str, columns={
        'days_up_sum': 'avg_days_up_user',
        'times_put_up': 'avg_times_up_user'
    })
gp.head()

n_user_items = all_samples.groupby(['user_id'])[['item_id']].count().reset_index() \
    .rename(index=str, columns={
        'item_id': 'n_user_items'
    })
gp = gp.merge(n_user_items, on='user_id', how='left')

gp.head()


train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')

train = train.merge(gp, on='user_id', how='left')
test = test.merge(gp, on='user_id', how='left')

agg_cols = list(gp.columns)[1:]

del gp
gc.collect()

train.head()

train = train[['avg_days_up_user','avg_times_up_user','n_user_items']]
test = test[['avg_days_up_user','avg_times_up_user','n_user_items']]

train.to_feather('../train_kag_agg_ftr.ftr')
test.to_feather('../test_kag_agg_ftr.ftr')

def catEncode(train_char, test_char, y, colLst = [], nbag = 10, nfold = 20, minCount = 3, postfix = ''):
    train_df = train_char.copy()
    test_df = test_char.copy()
    if not colLst:
        print("Empty ColLst")
        for c in train_char.columns:
            data = train_char[[c]].copy()
            data['y'] = y
            enc_mat = np.zeros((y.shape[0],4))
            enc_mat_test = np.zeros((test_char.shape[0],4))
            for bag in np.arange(nbag):
                kf = model_selection.KFold(n_splits= nfold, shuffle=True, random_state=2017*bag)
                for dev_index, val_index in kf.split(range(data['y'].shape[0])):
                    dev_X, val_X = data.iloc[dev_index,:], data.iloc[val_index,:]
                    datax = dev_X.groupby([c]).agg([len,np.mean,np.std, np.median])
                    datax.columns = ['_'.join(col).strip() for col in datax.columns.values]
#                    datax = datax.loc[datax.y_len > minCount]
                    ind = c + postfix
                    datax.rename(columns = {'y_mean': ('y_mean_' + ind), 'y_std': ('y_std_' + ind),
                    'y_len_': ('y_len' + ind), 'y_median_': ('y_median' + ind),}, inplace = True)
#                    datax[c+'_medshftenc'] =  datax['y_median']-med_y
#                    datax.drop(['y_len','y_mean','y_std','y_median'],axis=1,inplace=True)
                    datatst = test_char[[c]].copy()
                    val_X = val_X.join(datax,on=[c], how='left').fillna(np.mean(y))
                    datatst = datatst.join(datax,on=[c], how='left').fillna(np.mean(y))
                    enc_mat[val_index,...] += val_X[list(set(datax.columns)-set([c]))]
                    enc_mat_test += datatst[list(set(datax.columns)-set([c]))]
            enc_mat_test /= (nfold * nbag)
            enc_mat /= (nbag)        
            enc_mat = pd.DataFrame(enc_mat)  
            enc_mat.columns=[ind + str(x) for x in list(set(datax.columns)-set([c]))] 
            enc_mat_test = pd.DataFrame(enc_mat_test)  
            enc_mat_test.columns=enc_mat.columns
            train_df = pd.concat((enc_mat.reset_index(drop = True),train_df.reset_index(drop = True)), axis=1)
            test_df = pd.concat([enc_mat_test.reset_index(drop = True),test_df.reset_index(drop = True)],axis=1)
        
    else:
        print("Not Empty ColLst")
        data = train_char[colLst].copy()
        data['y'] = y
        enc_mat = np.zeros((y.shape[0],4))
        enc_mat_test = np.zeros((test_char.shape[0],4))
        for bag in np.arange(nbag):     
            kf = model_selection.KFold(n_splits= nfold, shuffle=True, random_state=2017*bag)
            for dev_index, val_index in kf.split(range(data['y'].shape[0])):
                dev_X, val_X = data.iloc[dev_index,:], data.iloc[val_index,:]
                datax = dev_X.groupby(colLst).agg([len,np.mean,np.std, np.median])
                datax.columns = ['_'.join(col).strip() for col in datax.columns.values]
#                datax = datax.loc[datax.y_len > minCount]
                ind = '_'.join(colLst) + postfix
                datax.rename(columns = {'y_mean': ('y_mean_' + ind), 'y_std': ('y_std_' + ind),
                'y_len': ('y_len_' + ind), 'y_median': ('y_median_' + ind),}, inplace = True)
                datatst = test_char[colLst].copy()
                val_X = val_X.join(datax,on=colLst, how='left').fillna(np.mean(y))
                datatst = datatst.join(datax,on=colLst, how='left').fillna(np.mean(y))
                print(val_X[list(set(datax.columns)-set(colLst))].columns)
                enc_mat[val_index,...] += val_X[list(set(datax.columns)-set(colLst))]
                enc_mat_test += datatst[list(set(datax.columns)-set(colLst))]
                
        enc_mat_test /= (nfold * nbag)
        enc_mat /= (nbag)        
        enc_mat = pd.DataFrame(enc_mat)  
        enc_mat.columns=[ind + str(x) for x in list(set(datax.columns)-set([c]))] 
        enc_mat_test = pd.DataFrame(enc_mat_test)  
        enc_mat_test.columns=enc_mat.columns 
        train_df = pd.concat((enc_mat.reset_index(drop = True),train_df.reset_index(drop = True)), axis=1)
        test_df = pd.concat([enc_mat_test.reset_index(drop = True),test_df.reset_index(drop = True)],axis=1)
    print(train_df.columns)
    print(test_df.columns)                            
    for c in train_df.columns:
        if train_df[c].dtype == 'float64':
            train_df[c] = train_df[c].astype('float32')
            test_df[c] = test_df[c].astype('float32')
    return train_df, test_df


train_df = pd.read_feather('../train_basic_features_woCats.pkl')
test_df = pd.read_feather('../test__basic_features_woCats.pkl')


train_df["item_seq_number"] = train_df['item_seq_number'].astype('int') // 10 * 10
test_df["item_seq_number"] = test_df['item_seq_number'].astype('int') // 10 * 10

train_df.loc[(train_df.item_seq_number >199) & (train_df.item_seq_number<=599),'item_seq_number'] = 599
train_df.loc[(train_df.item_seq_number >599) & (train_df.item_seq_number<=999),'item_seq_number'] = 999
train_df.loc[(train_df.item_seq_number >999),'item_seq_number' ] = 1999

test_df.loc[(test_df.item_seq_number >199) & (test_df.item_seq_number<=599),'item_seq_number'] = 599
test_df.loc[(test_df.item_seq_number >599) & (test_df.item_seq_number<=999),'item_seq_number'] = 999
test_df.loc[(test_df.item_seq_number >999) ,'item_seq_number'] = 1999


catCols = ['item_seq_number']

train_df, test_df = catEncode(train_df[catCols].copy(), test_df[catCols].copy(), train_df.deal_probability.values, nbag = 10, nfold = 10, minCount = 0)

train_df.drop(catCols, inplace = True, axis='columns')
test_df.drop(catCols, inplace = True, axis='columns')

catCols = ['item_seq_numbery_median']

train_df.drop(catCols, inplace = True, axis='columns')
test_df.drop(catCols, inplace = True, axis='columns')

train_df.to_feather('../train_itemseq_targetenc.pkl')
test_df.to_feather('../test_itemseq_targetenc.pkl')

train_df = pd.read_feather('../train_basic_features_woCats.pkl')
test_df = pd.read_feather('../test__basic_features_woCats.pkl')


train_df["image_top_1"] = train_df['image_top_1'].fillna(-1).astype('int') // 10 * 10
test_df["image_top_1"] = test_df['image_top_1'].fillna(-1).astype('int') // 10 * 10

catCols = ['image_top_1']

train_df, test_df = catEncode(train_df[catCols].copy(), test_df[catCols].copy(), train_df.deal_probability.values, nbag = 10, nfold = 10, minCount = 0)

train_df.drop(catCols, inplace = True, axis='columns')
test_df.drop(catCols, inplace = True, axis='columns')

train_df.to_feather('../train_imagetop_targetenc.pkl')
test_df.to_feather('../test_imagetop_targetenc.pkl')

