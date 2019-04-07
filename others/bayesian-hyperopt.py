import cPickle
import functools
from collections import defaultdict

import numpy as np
import pandas as pd
import xgboost as xgb

from nltk.corpus import stopwords
from collections import Counter
from sklearn.metrics import log_loss
from sklearn.cross_validation import train_test_split

from sklearn import model_selection, preprocessing, ensemble
from sklearn.metrics import log_loss
import gc
import math
import tqdm
from hyperopt import hp, fmin, tpe, STATUS_OK, Trials 

def word_match_share(row, stops=None):
    q1words = {}
    q2words = {}
    for word in row['question1']:
        if word not in stops:
            q1words[word] = 1
    for word in row['question2']:
        if word not in stops:
            q2words[word] = 1
    if len(q1words) == 0 or len(q2words) == 0:
        # The computer-generated chaff includes a few questions that are nothing but stopwords
        return 0
    shared_words_in_q1 = [w for w in q1words.keys() if w in q2words]
    shared_words_in_q2 = [w for w in q2words.keys() if w in q1words]
    R = (len(shared_words_in_q1) + len(shared_words_in_q2))/(len(q1words) + len(q2words))
    return R

def jaccard(row):
    wic = set(row['question1']).intersection(set(row['question2']))
    uw = set(row['question1']).union(row['question2'])
    if len(uw) == 0:
        uw = [1]
    return (len(wic) / len(uw))

def common_words(row):
    return len(set(row['question1']).intersection(set(row['question2'])))

def total_unique_words(row):
    return len(set(row['question1']).union(row['question2']))

def total_unq_words_stop(row, stops):
    return len([x for x in set(row['question1']).union(row['question2']) if x not in stops])

def wc_diff(row):
    return abs(len(row['question1']) - len(row['question2']))

def wc_ratio(row):
    l1 = len(row['question1'])*1.0 
    l2 = len(row['question2'])
    if l2 == 0:
        return np.nan
    if l1 / l2:
        return l2 / l1
    else:
        return l1 / l2

def wc_diff_unique(row):
    return abs(len(set(row['question1'])) - len(set(row['question2'])))

def wc_ratio_unique(row):
    l1 = len(set(row['question1'])) * 1.0
    l2 = len(set(row['question2']))
    if l2 == 0:
        return np.nan
    if l1 / l2:
        return l2 / l1
    else:
        return l1 / l2

def wc_diff_unique_stop(row, stops=None):
    return abs(len([x for x in set(row['question1']) if x not in stops]) - len([x for x in set(row['question2']) if x not in stops]))

def wc_ratio_unique_stop(row, stops=None):
    l1 = len([x for x in set(row['question1']) if x not in stops])*1.0 
    l2 = len([x for x in set(row['question2']) if x not in stops])
    if l2 == 0:
        return np.nan
    if l1 / l2:
        return l2 / l1
    else:
        return l1 / l2

def same_start_word(row):
    if not row['question1'] or not row['question2']:
        return np.nan
    return int(row['question1'][0] == row['question2'][0])

def char_diff(row):
    return abs(len(''.join(row['question1'])) - len(''.join(row['question2'])))

def char_ratio(row):
    l1 = len(''.join(row['question1'])) 
    l2 = len(''.join(row['question2']))
    if l2 == 0:
        return np.nan
    if l1 / l2:
        return l2 / l1
    else:
        return l1 / l2

def char_diff_unique_stop(row, stops=None):
    return abs(len(''.join([x for x in set(row['question1']) if x not in stops])) - len(''.join([x for x in set(row['question2']) if x not in stops])))


def get_weight(count, eps=10000, min_count=2):
    if count < min_count:
        return 0
    else:
        return 1 / (count + eps)
    
def tfidf_word_match_share_stops(row, stops=None, weights=None):
    q1words = {}
    q2words = {}
    for word in row['question1']:
        if word not in stops:
            q1words[word] = 1
    for word in row['question2']:
        if word not in stops:
            q2words[word] = 1
    if len(q1words) == 0 or len(q2words) == 0:
        # The computer-generated chaff includes a few questions that are nothing but stopwords
        return 0
    
    shared_weights = [weights.get(w, 0) for w in q1words.keys() if w in q2words] + [weights.get(w, 0) for w in q2words.keys() if w in q1words]
    total_weights = [weights.get(w, 0) for w in q1words] + [weights.get(w, 0) for w in q2words]
    
    R = np.sum(shared_weights) / np.sum(total_weights)
    return R

def tfidf_word_match_share(row, weights=None):
    q1words = {}
    q2words = {}
    for word in row['question1']:
        q1words[word] = 1
    for word in row['question2']:
        q2words[word] = 1
    if len(q1words) == 0 or len(q2words) == 0:
        # The computer-generated chaff includes a few questions that are nothing but stopwords
        return 0
    
    shared_weights = [weights.get(w, 0) for w in q1words.keys() if w in q2words] + [weights.get(w, 0) for w in q2words.keys() if w in q1words]
    total_weights = [weights.get(w, 0) for w in q1words] + [weights.get(w, 0) for w in q2words]
    
    R = np.sum(shared_weights) / np.sum(total_weights)
    return R


def build_features(data, stops, weights):
    X = pd.DataFrame()
    f = functools.partial(word_match_share, stops=stops)
    X['word_match'] = data.apply(f, axis=1, raw=True) #1

    f = functools.partial(tfidf_word_match_share, weights=weights)
    X['tfidf_wm'] = data.apply(f, axis=1, raw=True) #2

    f = functools.partial(tfidf_word_match_share_stops, stops=stops, weights=weights)
    X['tfidf_wm_stops'] = data.apply(f, axis=1, raw=True) #3

    X['jaccard'] = data.apply(jaccard, axis=1, raw=True) #4
    X['wc_diff'] = data.apply(wc_diff, axis=1, raw=True) #5
    X['wc_ratio'] = data.apply(wc_ratio, axis=1, raw=True) #6
    X['wc_diff_unique'] = data.apply(wc_diff_unique, axis=1, raw=True) #7
    X['wc_ratio_unique'] = data.apply(wc_ratio_unique, axis=1, raw=True) #8

    f = functools.partial(wc_diff_unique_stop, stops=stops)    
    X['wc_diff_unq_stop'] = data.apply(f, axis=1, raw=True) #9
    f = functools.partial(wc_ratio_unique_stop, stops=stops)    
    X['wc_ratio_unique_stop'] = data.apply(f, axis=1, raw=True) #10

    X['same_start'] = data.apply(same_start_word, axis=1, raw=True) #11
    X['char_diff'] = data.apply(char_diff, axis=1, raw=True) #12

    f = functools.partial(char_diff_unique_stop, stops=stops) 
    X['char_diff_unq_stop'] = data.apply(f, axis=1, raw=True) #13

#     X['common_words'] = data.apply(common_words, axis=1, raw=True)  #14
    X['total_unique_words'] = data.apply(total_unique_words, axis=1, raw=True)  #15

    f = functools.partial(total_unq_words_stop, stops=stops)
    X['total_unq_words_stop'] = data.apply(f, axis=1, raw=True)  #16
    
    X['char_ratio'] = data.apply(char_ratio, axis=1, raw=True) #17    

    return X


#if __name__ == '__main__':
df_train = pd.read_csv('../data/train.csv')
df_train = df_train.fillna(' ')

df_test = pd.read_csv('../data/test.csv')
ques = pd.concat([df_train[['question1', 'question2']], \
    df_test[['question1', 'question2']]], axis=0).reset_index(drop='index')
q_dict = defaultdict(set)
for i in range(ques.shape[0]):
        q_dict[ques.question1[i]].add(ques.question2[i])
        q_dict[ques.question2[i]].add(ques.question1[i])

def q1_freq(row):
    return(len(q_dict[row['question1']]))
    
def q2_freq(row):
    return(len(q_dict[row['question2']]))
    
def q1_q2_intersect(row):
    return(len(set(q_dict[row['question1']]).intersection(set(q_dict[row['question2']]))))

df_train['q1_q2_intersect'] = df_train.apply(q1_q2_intersect, axis=1, raw=True)
df_train['q1_freq'] = df_train.apply(q1_freq, axis=1, raw=True)
df_train['q2_freq'] = df_train.apply(q2_freq, axis=1, raw=True)

df_test['q1_q2_intersect'] = df_test.apply(q1_q2_intersect, axis=1, raw=True)
df_test['q1_freq'] = df_test.apply(q1_freq, axis=1, raw=True)
df_test['q2_freq'] = df_test.apply(q2_freq, axis=1, raw=True)

test_leaky = df_test.loc[:, ['q1_q2_intersect','q1_freq','q2_freq']]
del df_test

train_leaky = df_train.loc[:, ['q1_q2_intersect','q1_freq','q2_freq']]
train_leaky.to_csv('train_leaky.csv',index=False)
test_leaky.to_csv('test_leaky.csv',index=False)
# explore
stops = set(stopwords.words("english"))

df_train['question1'] = df_train['question1'].map(lambda x: str(x).lower().split())
df_train['question2'] = df_train['question2'].map(lambda x: str(x).lower().split())

train_qs = pd.Series(df_train['question1'].tolist() + df_train['question2'].tolist())

words = [x for y in train_qs for x in y]
counts = Counter(words)
weights = {word: get_weight(count) for word, count in counts.items()}

print('Building Features')
X_train = build_features(df_train, stops, weights)
X_train.to_csv('kaggle_ftr.csv',index=False)
######################
print('Building Whole Datasets')
df_train = pd.read_csv('../data/train.csv')
df_train = df_train.fillna(' ')

X_train = pd.read_csv('kaggle_ftr.csv')
train = pd.read_csv('../data/train_save_all_ftr_wo_vec.csv')
with open('../data/more_magic_train.pkl', "rb") as f:
    train_df = cPickle.load(f)
train = train.set_index('test_id').join(train_df.set_index('test_id'),rsuffix="_2").reset_index()
features=[
   'len_q1', 'len_q2', 'diff_len',
   'len_char_q1', 'len_char_q2', 'len_word_q1', 'len_word_q2',
   'common_words', 'fuzz_qratio', 'fuzz_WRatio', 'fuzz_partial_ratio',
   'fuzz_partial_token_set_ratio', 'fuzz_partial_token_sort_ratio',
   'fuzz_token_set_ratio', 'fuzz_token_sort_ratio', 'wmd', 'norm_wmd',
   'cosine_distance', 'cityblock_distance', #'jaccard_distance',
   'canberra_distance', #'euclidean_distance', 
   'minkowski_distance',
   'braycurtis_distance', 'skew_q1vec', 'skew_q2vec', 'kur_q1vec',
   'kur_q2vec',
   'count_of_question1_unigram',
   'count_of_unique_question1_unigram',
   'ratio_of_unique_question1_unigram',
   'count_of_digit_in_question1_unigram',
   'ratio_of_digit_in_question1_unigram', 
   'count_of_question2_unigram',
   'count_of_unique_question2_unigram',
   'ratio_of_unique_question2_unigram',
   'count_of_digit_in_question2_unigram',
   'ratio_of_digit_in_question2_unigram', 'count_of_question1_bigram',
   'count_of_unique_question1_bigram',
   'ratio_of_unique_question1_bigram',
   'count_of_digit_in_question1_bigram',
   'ratio_of_digit_in_question1_bigram', 'count_of_question2_bigram',
   'count_of_unique_question2_bigram',
   'ratio_of_unique_question2_bigram', 'count_of_question1_trigram',
   'count_of_unique_question1_trigram',
   'ratio_of_unique_question1_trigram',
   'count_of_digit_in_question1_trigram',
   'ratio_of_digit_in_question1_trigram', 'count_of_question2_trigram',
   'count_of_unique_question2_trigram',
   'ratio_of_unique_question2_trigram',
   'count_of_question1_unigram_in_question2',
   'ratio_of_question1_unigram_in_question2',
   'question1_unigram_div_question2_unigram',
   'count_of_question2_unigram_in_question1',
   'ratio_of_question2_unigram_in_question1',
   'question2_unigram_div_question1_unigram',
   'count_of_question1_bigram_in_question2',
   'ratio_of_question1_bigram_in_question2',
   'question1_bigram_div_question2_bigram',
   'count_of_question2_bigram_in_question1',
   'ratio_of_question2_bigram_in_question1',
   'question2_bigram_div_question1_bigram',
   'pos_of_question2_unigram_in_question1_min',
   'pos_of_question2_unigram_in_question1_mean',
   'pos_of_question2_unigram_in_question1_median',
   'pos_of_question2_unigram_in_question1_max',
   'pos_of_question2_unigram_in_question1_std',
   'normalized_pos_of_question2_unigram_in_question1_min',
   'normalized_pos_of_question2_unigram_in_question1_mean',
   'normalized_pos_of_question2_unigram_in_question1_median',
   'normalized_pos_of_question2_unigram_in_question1_max',
   'normalized_pos_of_question2_unigram_in_question1_std',
   'pos_of_question1_unigram_in_question2_min',
   'pos_of_question1_unigram_in_question2_mean',
   'pos_of_question1_unigram_in_question2_median',
   'pos_of_question1_unigram_in_question2_max',
   'pos_of_question1_unigram_in_question2_std',
   'normalized_pos_of_question1_unigram_in_question2_min',
   'normalized_pos_of_question1_unigram_in_question2_mean',
   'normalized_pos_of_question1_unigram_in_question2_median',
   'normalized_pos_of_question1_unigram_in_question2_max',
   'normalized_pos_of_question1_unigram_in_question2_std',
   'pos_of_question2_bigram_in_question1_min',
   'pos_of_question2_bigram_in_question1_mean',
   'pos_of_question2_bigram_in_question1_median',
   'pos_of_question2_bigram_in_question1_max',
   'pos_of_question2_bigram_in_question1_std',
   'normalized_pos_of_question2_bigram_in_question1_min',
   'normalized_pos_of_question2_bigram_in_question1_mean',
   'normalized_pos_of_question2_bigram_in_question1_median',
   'normalized_pos_of_question2_bigram_in_question1_max',
   'normalized_pos_of_question2_bigram_in_question1_std',
   'pos_of_question1_bigram_in_question2_min',
   'pos_of_question1_bigram_in_question2_mean',
   'pos_of_question1_bigram_in_question2_median',
   'pos_of_question1_bigram_in_question2_max',
   'pos_of_question1_bigram_in_question2_std',
   'normalized_pos_of_question1_bigram_in_question2_min',
   'normalized_pos_of_question1_bigram_in_question2_mean',
   'normalized_pos_of_question1_bigram_in_question2_median',
   'normalized_pos_of_question1_bigram_in_question2_max',
   'normalized_pos_of_question1_bigram_in_question2_std',
   'jaccard_coef_of_unigram_between_question1_question2',
   'jaccard_coef_of_bigram_between_question1_question2',
   'jaccard_coef_of_trigram_between_question1_question2',
   'dice_dist_of_unigram_between_question1_question2',
   'dice_dist_of_bigram_between_question1_question2',
   'dice_dist_of_trigram_between_question1_question2',
   'count_of_question1_biterm', 'count_of_unique_question1_biterm',
   'ratio_of_unique_question1_biterm', 'count_of_question2_biterm',
   'count_of_unique_question2_biterm',
   'ratio_of_unique_question2_biterm',
   'count_of_question1_biterm_in_question2',
   'ratio_of_question1_biterm_in_question2',
   'question1_biterm_div_question2_biterm',
   'count_of_question2_biterm_in_question1',
   'ratio_of_question2_biterm_in_question1',
   'question2_biterm_div_question1_biterm',
   'pos_of_question2_biterm_in_question1_min',
   'pos_of_question2_biterm_in_question1_mean',
   'pos_of_question2_biterm_in_question1_median',
   'pos_of_question2_biterm_in_question1_max',
   'pos_of_question2_biterm_in_question1_std',
   'normalized_pos_of_question2_biterm_in_question1_min',
   'normalized_pos_of_question2_biterm_in_question1_mean',
   'normalized_pos_of_question2_biterm_in_question1_median',
   'normalized_pos_of_question2_biterm_in_question1_max',
   'normalized_pos_of_question2_biterm_in_question1_std',
   'pos_of_question1_biterm_in_question2_min',
   'pos_of_question1_biterm_in_question2_mean',
   'pos_of_question1_biterm_in_question2_median',
   'pos_of_question1_biterm_in_question2_max',
   'pos_of_question1_biterm_in_question2_std',
   'normalized_pos_of_question1_biterm_in_question2_min',
   'normalized_pos_of_question1_biterm_in_question2_mean',
   'normalized_pos_of_question1_biterm_in_question2_median',
   'normalized_pos_of_question1_biterm_in_question2_max',
   'normalized_pos_of_question1_biterm_in_question2_std',
   'jaccard_coef_of_biterm_between_question1_question2',
   'dice_dist_of_biterm_between_question1_question2',
   'cosine_distance_tfidf',
   'cityblock_distance_tfidf', 'jaccard_distance_tfidf',
   'canberra_distance_tfidf', 'euclidean_distance_tfidf',
   'minkowski_distance_tfidf', 'braycurtis_distance_tfidf',
   'skew_q1vec_tfidf', 'skew_q2vec_tfidf', 'kur_q1vec_tfidf',
   'kur_q2vec_tfidf',
    ]
 
X_train_ab = train[features]

########
train_leaky=pd.read_csv('train_leaky.csv')
test_leaky=pd.read_csv('test_leaky.csv')

########
train = pd.concat((X_train, X_train_ab, train_leaky), axis=1)
n_train = X_train.shape[0]

#    X_train['test_id'] = df_train['id']
train_y = df_train['is_duplicate'].values
del(X_train,train_df,df_train,X_train_ab, train_leaky)
gc.collect()
print("Split Data to Training and Valid...")
i=0
nfold = 5
oob_valpred = np.zeros((n_train,1))
modelLst =[]
kf = model_selection.KFold(n_splits=nfold, shuffle=True, random_state=201705)
for dev_index, val_index in kf.split(range(n_train)):
    X_train, X_valid = train.loc[dev_index], train.loc[val_index]
    y_train, y_valid = train_y[dev_index], train_y[val_index]

    #UPDownSampling
    pos_train = X_train[y_train == 1]
    neg_train = X_train[y_train == 0]
    X_train = pd.concat((neg_train, pos_train.iloc[:int(0.8*len(pos_train))], neg_train))
    y_train = np.array([0] * neg_train.shape[0] + [1] * pos_train.iloc[:int(0.8*len(pos_train))].shape[0] + [0] * neg_train.shape[0])
    print(np.mean(y_train))
    del pos_train, neg_train

    pos_valid = X_valid[y_valid == 1]
    neg_valid = X_valid[y_valid == 0]
    X_valid = pd.concat((neg_valid, pos_valid.iloc[:int(0.8 * len(pos_valid))], neg_valid))
    y_valid = np.array([0] * neg_valid.shape[0] + [1] * pos_valid.iloc[:int(0.8 * len(pos_valid))].shape[0] + [0] * neg_valid.shape[0])
    print(np.mean(y_valid))
    del pos_valid, neg_valid


    params = {}
    params['objective'] = 'binary:logistic'
    params['eval_metric'] = 'logloss'
    params['eta'] = 0.02
    params['max_depth'] = 7
    params['subsample'] = 0.6
    params['base_score'] = 0.2
    # params['scale_pos_weight'] = 0.2

    d_train = xgb.DMatrix(X_train, label=y_train)
    d_valid = xgb.DMatrix(X_valid, label=y_valid)

    watchlist = [(d_train, 'train'), (d_valid, 'valid')]

    model = xgb.train(params, d_train, 502, watchlist, early_stopping_rounds=50, verbose_eval=1)
    best_iteration = model.best_iteration+1
    model = xgb.train(params, d_train, best_iteration, watchlist, early_stopping_rounds=50, verbose_eval=1)
    modelLst.append(model)
    print(log_loss(y_valid, model.predict(d_valid)))
    preds = model.predict(d_valid)
    oob_valpred[val_index,...] = preds.reshape(-1,1)
#    model.save_model(args.save + '.mdl')

train_cols = X_train.columns
del(X_train,X_valid,y_train,y_valid)
gc.collect()
print('Building Test Features')
test = pd.read_csv('../data/test_save_all_ftr_wo_vec.csv')    
with open('../data/more_magic_test.pkl', "rb") as f:
    test_df = cPickle.load(f)
test = test.set_index('test_id').join(test_df.set_index('test_id'),rsuffix="_2").reset_index()
X_test_ab = test[features]

df_test = pd.read_csv('../data/test.csv')
df_test = df_test.fillna(' ')

df_test['question1'] = df_test['question1'].map(lambda x: str(x).lower().split())
df_test['question2'] = df_test['question2'].map(lambda x: str(x).lower().split())

x_test = build_features(df_test, stops, weights)
x_test.to_csv('kaggle_ftr_test.csv',index=False)

x_test = pd.concat((x_test, x_test_ab, test_leaky), axis=1)
d_test = xgb.DMatrix(x_test)
p_test = model.predict(d_test)
sub = pd.DataFrame()
sub['test_id'] = df_test['test_id']
sub['is_duplicate'] = p_test
#sub.to_csv('../results/' + args.save + '.csv')
sub.to_csv('../results/' + 'leaky' + '.csv')


#####Tune
space = {
    "booster": "gbtree",
    "objective": 'binary:logistic',
    "scale_pos_weight": pos_wt[0],
    #"n_estimators" : hp.quniform("n_estimators", 500, 10000, 100),
    "learning_rate" : 0.1,#hp.qloguniform("learning_rate", 0.01, 0.1, 0.01),
    "gamma": hp.quniform("gamma",  0.0, 3.0, 0.1),
    "alpha" : hp.quniform("reg_alpha", 0.0, 3.0, 0.1),
    "lambda" : hp.quniform("reg_lambda",  0.0, 3.0,0.1),
    "min_child_weight": hp.quniform("min_child_weight", 0., 30, 0.2),
    "max_depth": hp.choice("max_depth", np.arange(3, 20,  dtype=int)),
    "subsample": hp.quniform("subsample", 0.3, 1.0, 0.05),
    "colsample_bytree": hp.quniform("colsample_bytree", 0.3, 1.0, 0.05),
    "colsample_bylevel": hp.quniform("colsample_bylevel", 0.1, 1, 0.05),
    "nthread": 2,
    "seed": 12345,#hp.quniform("n_estimators", [1,2017,12345,1695,23,54]),
    "eval_metric": "logloss"
}
def objective(space):

    plst = list(space.items())
    watchlist = [ (xgtrain,'train'), (xgval, 'val') ]    
    clf = xgb.train(plst, xgtrain, 150, watchlist, early_stopping_rounds=15)

    watchlist = [ (xgtrain,'train'), (xgval, 'val') ]
    preds = clf.predict(xgval)
    loss = log_loss(val_y, preds)
    print "SCORE:", loss

    return{'loss':loss, 'status': STATUS_OK }


trials = Trials()
best = fmin(fn=objective,
            space=space,
            algo=tpe.suggest,
            max_evals=100,
            trials=trials)

print best
#{'reg_alpha': 1.1, 'colsample_bylevel': 0.75, 'min_child_weight': 1.0, 'subsample': 0.9500000000000001, 'reg_lambda': 1.7000000000000002, 'max_depth': 13, 'gamma': 0.2, 'colsample_bytree' : 1}
#{'reg_alpha': 0.4, 'colsample_bytree': 0.55, 'colsample_bylevel': 0.30000000000000004, 'min_child_weight': 2.6, 'subsample': 0.9500000000000001, 'reg_lambda': 0.0, 'max_depth': 13, 'gamma': 0.6000000000000001}

#Feature imp
###########################
from matplotlib import pylab as plt
import operator

def create_feature_map(features):
    outfile = open('xgb.fmap', 'w')
    i = 0
    for feat in features:
        outfile.write('{0}\t{1}\tq\n'.format(i, feat))
        i = i + 1

    outfile.close()
    
create_feature_map(X_train.columns)
importance = model.get_fscore(fmap='xgb.fmap')
importance = sorted(importance.items(), key=operator.itemgetter(1))

df = pd.DataFrame(importance, columns=['feature', 'fscore'])
df['fscore'] = df['fscore'] / df['fscore'].sum()

plt.figure()
df.plot()
df.plot(kind='barh', x='feature', y='fscore', legend=False, figsize=(6, 10))
plt.title('XGBoost Feature Importance')
plt.xlabel('relative importance')
plt.gcf().savefig('feature_importance_xgb.png')    

df.sort(['feature'],ascending=False)
df

#Bayesian OPT
import datetime
from bayes_opt import BayesianOptimization
from sklearn import metrics

def xgboostcv(max_depth,subsample,colsample_bytree,gamma,
              min_child_weight,alpha,reg_lambda,colsample_bylevel,seed=201705,eval_metric="logloss",nthread=3,
              n_estimators=150,#scale_pos_weight=pos_wt[0],
              objective='binary:logistic',booster="gbtree",learning_rate=0.02):

    num_rounds =n_estimators
    param = {}
    param['objective'] = objective
    param['eta'] = learning_rate
    param['max_depth'] = int(max_depth)
    param['silent'] = 1
    param['eval_metric'] = eval_metric
    param['min_child_weight'] = min_child_weight
    param['subsample'] = subsample
    param['colsample_bytree'] = colsample_bytree
    param['seed'] = seed
    param['nthread'] = nthread
    #param['scale_pos_weight'] = scale_pos_weight
    param['base_score'] = 0.17
    #param['tree_method'] = 'approx'#'exact'#/
    param['alpha'] = alpha
    param['lambda'] = reg_lambda
    param['gamma'] = gamma
    param['colsample_bylevel'] = colsample_bylevel
    plst = list(param.items())
    watchlist = [ (d_train,'train'), (d_valid, 'val') ]    
    clf = xgb.train(plst, d_train, num_rounds, watchlist, early_stopping_rounds=15)

    watchlist = [ (d_train,'train'), (d_valid, 'val') ]
    preds = clf.predict(d_valid)
    loss = log_loss(y_valid, preds)
    print "SCORE:", loss
    return -1.0*loss
	
xgboostBO = BayesianOptimization(xgboostcv,
								 {'max_depth': (int(2), int(20)),
								  #'learning_rate': (0.01, 0.03),
								  #'n_estimators': (int(1000), int(20000)),
								  'subsample': (0.1, 1.0),
								  'colsample_bytree': (0.1, 1.0),
								  'gamma': (0.0, 1.0),
								  'min_child_weight': (0, 10.0),
								  'alpha': (0.0,5.0),
								  'reg_lambda': (0.0,5.0),
								  'colsample_bylevel': (0.1, 1.0) 
								 })

xgboostBO.maximize(init_points=5, n_iter=25, acq='ei')
print('-'*53)

print('Final Results')
print('XGBOOST: %f' % xgboostBO.res['max']['max_val'])	
print(xgboostBO.res['max'])
print(xgboostBO.res['all'])
##########################
Initialization
----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
 Step |   Time |      Value |     alpha |   colsample_bylevel |   colsample_bytree |     gamma |      max_depth |   min_child_weight |   reg_lambda |   subsample |
