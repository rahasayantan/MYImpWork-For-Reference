import cPickle
import os
import sys
import scipy as sc
import operator
import numpy as np
import pandas as pd
from scipy import sparse
import xgboost as xgb
from sklearn import model_selection, preprocessing, ensemble
from sklearn.metrics import log_loss
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from bs4 import BeautifulSoup
#reload(sys)
#sys.setdefaultencoding('utf8')
#r = re.compile(r"\s")
from sklearn.preprocessing import LabelEncoder
import re
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import TruncatedSVD
import nltk
from scipy.stats import boxcox
from sklearn.decomposition import TruncatedSVD
import datetime as dt
from nltk.stem.porter import *
import gc
import math
from collections import Counter

nfold = 5
    
###################################    
data_path = "../input/"
train_file = data_path + "train.json"
test_file = data_path + "test.json"
train_df = pd.read_json(train_file)
test_df = pd.read_json(test_file)

print(train_df.shape)
print(test_df.shape)



# Columns
#Index([u'bathrooms', u'bedrooms', u'building_id', u'created', u'description',
#       u'display_address', u'features', u'interest_level', u'latitude',
#       u'listing_id', u'longitude', u'manager_id', u'photos', u'price',
#       u'street_address'],
#      dtype='object')
#

target_col='interest_level'
target_num_map = {'high':0, 'medium':1, 'low':2}
train_y = np.array(train_df['interest_level'].apply(lambda x: target_num_map[x]))

ntrain = train_df.shape[0]
train_df = train_df.drop([ 'interest_level'], axis=1)
test_df_listing_id = test_df.listing_id.values

train_test = pd.concat((train_df, test_df), axis=0).reset_index(drop=True)

# count of photos #
train_test["num_photos"] = train_test["photos"].apply(len)

# count of "features" #
train_test["num_features"] = train_test["features"].apply(len)

# convert the created column to datetime object so as to extract more features 
train_test["created"] = pd.to_datetime(train_test["created"])

# Let us extract some features like year, month, day, hour from date columns #
train_test["created_year"] = train_test["created"].dt.year
train_test["created_month"] = train_test["created"].dt.month
train_test["created_day"] = train_test["created"].dt.day
train_test["created_hour"] = train_test["created"].dt.hour
train_test['created_weekday'] = train_test['created'].dt.dayofweek
train_test['created_wd'] = ((train_test['created_weekday'] != 5) & (train_test['created_weekday'] != 6)).astype(int)
train_test['created'] = train_test['created'].map(lambda x: float((x - dt.datetime(2015, 12, 30)).days) + (float((x - dt.datetime(2015, 12, 30)).seconds) / 86400))

#drop created_year
#train_test.drop('created_year',inplace=True,axis=1)

bc_price, tmp = boxcox(train_test.price)
lg_price = np.log1p(train_test.price)
train_test['bc_price'] = bc_price
train_test['lg_price'] = lg_price
train_test['Zero_building_id'] = train_test['building_id'].apply(lambda x: 1 if x == '0' else 0)
train_test['Zero_Ftr'] = train_test['features'].apply(lambda x: 1 if x == [] else 0)
train_test['Zero_description'] = train_test['description'].apply(lambda x: 1 if x == '' else 0)

def cleanhtml(raw_html):
    cleanr = re.compile('<.*?>')
    cleantext = re.sub(cleanr, ' ', raw_html)
    cleanr = re.compile('\\r')
    cleantext = re.sub(cleanr, ' ', cleantext)
    cleantext = BeautifulSoup(cleantext,"lxml").get_text(separator=" ")
    cleantext = re.sub(r'[^a-zA-Z0-9\s]', ' ', cleantext)
    stopwords = nltk.corpus.stopwords.words("english")
    stopwords = set(stopwords)

    cleantext = " ".join([w for w in cleantext.split(' ') if w not in stopwords])
    ##use stemmer
    stemmer = PorterStemmer()
    cleantext = (" ").join([stemmer.stem(z) for z in cleantext.split(" ")])
    return cleantext.lower().strip()

#train_df.loc[train_df.building_id=='ff99dfa1943c3c0ce7510e72bb86d32f','description']
train_test["description_orig"] = train_test["description"]

train_test["description"] = train_test["description"].apply(cleanhtml)
train_test['description'] = train_test['description'].apply(lambda x: x.replace('<p><a  website_redacted ', ''))

train_test["num_description_words"] = train_test["description"].apply(lambda x:0 if x == '' else len(x.split(" ")))
train_test["ratio_description_words"] = train_test["num_description_words"]/train_test["num_description_words"].mean()
train_test["ratio_description_words"]  =  train_test["ratio_description_words"].apply(lambda x: round(x,2))
#################
########################
####Count of Text Features
########################
##From Chen's Crowdflower
def getUnigram(words):
    """
        Input: a list of words, e.g., ['I', 'am', 'Denny']
        Output: a list of unigram
    """
    assert type(words) == list
    return words
def getBigram(words, join_string, skip=0):
	"""
	   Input: a list of words, e.g., ['I', 'am', 'Denny']
	   Output: a list of bigram, e.g., ['I_am', 'am_Denny']
	   I use _ as join_string for this example.
	"""
	assert type(words) == list
	L = len(words)
	if L > 1:
		lst = []
		for i in range(L-1):
			for k in range(1,skip+2):
				if i+k < L:
					lst.append( join_string.join([words[i], words[i+k]]) )
	else:
		# set it as unigram
		lst = getUnigram(words)
	return lst
def getTrigram(words, join_string, skip=0):
	"""
	   Input: a list of words, e.g., ['I', 'am', 'Denny']
	   Output: a list of trigram, e.g., ['I_am_Denny']
	   I use _ as join_string for this example.
	"""
	assert type(words) == list
	L = len(words)
	if L > 2:
		lst = []
		for i in range(L-2):
			for k1 in range(1,skip+2):
				for k2 in range(1,skip+2):
					if i+k1 < L and i+k1+k2 < L:
						lst.append( join_string.join([words[i], words[i+k1], words[i+k1+k2]]) )
	else:
		# set it as bigram
		lst = getBigram(words, join_string, skip)
	return lst    
def getFourgram(words, join_string):
    """
        Input: a list of words, e.g., ['I', 'am', 'Denny', 'boy']
        Output: a list of trigram, e.g., ['I_am_Denny_boy']
        I use _ as join_string for this example.
    """
    assert type(words) == list
    L = len(words)
    if L > 3:
        lst = []
        for i in xrange(L-3):
            lst.append( join_string.join([words[i], words[i+1], words[i+2], words[i+3]]) )
    else:
        # set it as bigram
        lst = getTrigram(words, join_string)
    return lst
def try_divide(x, y, val=0.0):
    """ 
    	Try to divide two numbers
    """
    if y != 0.0:
    	val = float(x) / y
    return val
    
train_test["bigram_description"] = train_test["description"].apply(lambda x: getBigram(x.split(' '),'_'))
train_test["trigram_description"] = train_test["description"].apply(lambda x: getTrigram(x.split(' '),'_'))
train_test["frgram_description"] = train_test["description"].apply(lambda x: getFourgram(x.split(' '),'_'))

train_test["bigram_cnt"] = train_test["bigram_description"].apply(lambda x: 0 if x==[''] else len(x))
train_test["trigram_cnt"] = train_test["trigram_description"].apply(lambda x: 0 if x==[''] else len(x))
train_test["frgram_cnt"] = train_test["frgram_description"].apply(lambda x: 0 if x==[''] else len(x))

train_test["uni_unigram_cnt"] = train_test["description"].apply(lambda x: 0 if x=='' else len(set(x)))
train_test["uni_bigram_cnt"] = train_test["bigram_description"].apply(lambda x: 0 if x==[''] else len(set(x)))
train_test["uni_trigram_cnt"] = train_test["trigram_description"].apply(lambda x: 0 if x==[''] else len(set(x)))
train_test["uni_frgram_cnt"] = train_test["frgram_description"].apply(lambda x: 0 if x==[''] else len(set(x)))
train_test["rat_unigram_cnt"] = map(try_divide,train_test["num_description_words"],train_test["uni_unigram_cnt"])
train_test["rat_bigram_cnt"] = map(try_divide,train_test["bigram_cnt"],train_test["uni_bigram_cnt"])
train_test["rat_trigram_cnt"] = map(try_divide,train_test["trigram_cnt"],train_test["uni_trigram_cnt"])
train_test["rat_frgram_cnt"] = map(try_divide,train_test["frgram_cnt"],train_test["uni_frgram_cnt"] )

count_digit = lambda x: sum([1. for w in x if w.isdigit()])
train_test["num_in_desc"] = train_test["description"].apply(lambda x: count_digit(x))
train_test["num_in_ftr"] = train_test["features"].apply(lambda x: count_digit(x))
train_test['desc_letters_count'] = train_test['description'].apply(lambda x: len(x.strip()))


####Feature Processing
train_test['features_0'] = train_test["features"].apply(lambda x: " ".join(["_".join(i.lower().split(" ")) for i in x]))

replace_ftr_dict = {
    "hr": "hour",
    "24hr": "24 hour",
    "24hour": "24 hour",
    "24/7": "24 hour",
    "24-hour":"24 hour",
    "a/c":"ac",
    "air conditioning":"ac",
    "dogs": "dog",
    "cats": "cat",
    "areas": "area",
    "roofdeck": "roof deck",
    "approval": "approv",
    "approved": "approv",
    "areas": "area",
    "apt.":"",
    "bathrooms": "bathroom",
    "baths": "bathroom",
    "bbqs": "bbq",
    "bedrooms": "bedroom",
    "blks": "blk",
    "block": "blk",
    "lounge":"room",
    "ceilings": "ceiling",
    "chefs": "chef",
    "childrens": "children",
    "cleaning": "clean",
    "closets": "closet",
    "closet\(s\)": "closet",
    "conditioning": "condition",
    "counters": "counter",
    "service":"",
    "decks": "deck",
    "concierge":"",
    "decorative":"deco",
    "eat in kitchen":"eatinkitchen",
    "eatin kitchen":"eatinkitchen",
    "elevator": "elev",
    "equipped": "equipment",
    "exposed": "exposure",
    "every floor":"floor",
    "facilities": "facility",
    "finishes": "finished",
    "fireplaces": "fireplace",
    "flooring": "floor",
    "floors": "floor",
    "glassenclosed":"glass enclosed",
    "countertops":"counter",
    "counter tops":"counter",
    "gardening": "garden",
    "garage parking":"garage",
    "gymfitness":"gym",
    "grilling": "grills",
    "highourise":  "high rise",
    "housekeeping service":"housekeeping",
    "swimming pool":"pool",
    "including": "included",
    "lighting": "light",
    "sized":"size",
    "livingroom":"living room",
    "lots": "lot",
    "live in":"livein",
    "supremeintendent":"supreme",
    "bathouroom":"bathroom",
    "screening room":" media room",
    "multi level":"multilevel",
    "nearby": "near",
    "natural light":"sunlight",
    "newly": "new",
    "okay"  : "ok",
    "parking lot":"parking",
    "pets":"pet",
    "pet friendly":"pet",
    "pet allowed":"pet",
    "pre-war":"prewar",
    "pre war":"prewar",
    "post war":"postwar",
    "parking available":"parking",
    "parking space":"parking",
    "renovated":"renovation",
    "renovations":"renovation",
    "rentable":"rent",
    "rental":"rent",
    "residents":"resident",
    "skylights":"skylight",
    "spaces":"space",
    "stabalized":"stabilized",
    "studios":"studio",
    "sundrenched": "sunlight",
    "sunfilled":"sunlight",
    "sunny":"sunlight",
    "super":"supreme",
    "ss appliances":"stainless",
    "tenants":"tenant",
    "terraces":"terrace",
    "training":"train",
    "units":"unit",
    "views":"view",
    "walls":"wall",
    "windows":"window"
    }

## replace other words
def clean(l):
    stopwords = nltk.corpus.stopwords.words("english")
    stopwords = set(stopwords)

    lf = []
    for c in l:
        for k,v in replace_ftr_dict.items():
            c = re.sub(k, v, c.lower())
        c = re.sub(r'[^a-zA-Z0-9\s]', '', c)
        lf.append(c.strip())
    c =[]
    for x in lf:
        c1 = ' '.join([w for w in x.split(' ') if w not in stopwords])
        c.append(c1)
    return(c)
    
train_test['features_1'] = train_test["features"].apply(clean)
train_test['features_2'] = train_test["features_1"].apply(lambda x: " ".join(["_".join(i.split(" ")) for i in x]))
train_test['features_3'] = train_test["features_1"].apply(lambda x: " ".join([" ".join(i.split(" ")) for i in x]))

stemmer = PorterStemmer()
train_test['features_stem'] = train_test["features_1"].apply(lambda x: " ".join([" ".join([stemmer.stem(z) for z in i.split(" ")]) for i in x]))

train_test["bigram_feature"] = train_test["features_stem"].apply(lambda x: getBigram(x.split(' '),'_'))
train_test["trigram_feature"] = train_test["features_stem"].apply(lambda x: getTrigram(x.split(' '),'_'))

##Get intersect Feature / Descriptiom
train_test['intersect_ftr_desc'] = list(train_test[['features_stem','description']].apply(lambda x: sum([1. for w in x['features_stem'].split(' ') if w in set(x['description'].split(' ')) ]),axis=1))

train_test['intersect_bigram_ftr_desc'] = list(train_test[['bigram_feature','bigram_description']].apply(lambda x: sum([1. for w in x['bigram_feature'] if w in set(x['bigram_description']) and w!='']),axis=1))

train_test['intersect_trigram_ftr_desc'] = list(train_test[['trigram_feature','trigram_description']].apply(lambda x: sum([1. for w in x['trigram_feature'] if w in set(x['trigram_description'])and w!='']),axis=1))

import gc
gc.collect()

def get_position_list(obs,target):
    """
        Get the list of positions of obs in target
    """
    pos_of_obs_in_target = [0]
    if len(obs) != 0:
        pos_of_obs_in_target = [j for j,w in enumerate(obs, start=1) if w in target]
        if len(pos_of_obs_in_target) == 0:
            pos_of_obs_in_target = [0]
    return pos_of_obs_in_target

###position of feature in description
pos = list(train_test[['features_stem','description']].apply(lambda x: get_position_list(x['features_stem'].split(' '),x['description'].split(' ')),axis=1))

## stats feat on pos
train_test["pos_of_features_stem_description_min"] = map(np.min, pos)
train_test["pos_of_features_stem_description_mean"] = map(np.mean, pos)
train_test["pos_of_features_stem_description_med"] = map(np.median, pos)
train_test["pos_of_features_stem_description_max"] = map(np.max, pos)
train_test["pos_of_features_stem_description_std"] = map(np.std, pos)
## stats feat on normalized_pos
train_test["norm_pos_of_features_stem_description_min"] = map(try_divide, train_test["pos_of_features_stem_description_min"], train_test["intersect_ftr_desc"])
train_test["norm_pos_of_features_stem_description_max"] = map(try_divide, train_test["pos_of_features_stem_description_max"], train_test["intersect_ftr_desc"])
train_test["norm_pos_of_features_stem_description_mean"] = map(try_divide, train_test["pos_of_features_stem_description_mean"], train_test["intersect_ftr_desc"])
train_test["norm_pos_of_features_stem_description_median"] = map(try_divide, train_test["pos_of_features_stem_description_max"], train_test["intersect_ftr_desc"])
train_test["norm_pos_of_features_stem_description_std"] = map(try_divide, train_test["pos_of_features_stem_description_std"] , train_test["intersect_ftr_desc"])

train_test["pos_of_features_stem_description_min2"] = map(np.min, pos)
train_test["pos_of_features_stem_description_mean2"] = map(np.mean, pos)
train_test["pos_of_features_stem_description_med2"] = map(np.median, pos)
train_test["pos_of_features_stem_description_max2"] = map(np.max, pos)
train_test["pos_of_features_stem_description_std2"] = map(np.std, pos)

train_test["norm_pos_of_features_stem_description_min2"] = map(try_divide, train_test["pos_of_features_stem_description_min2"], train_test["intersect_bigram_ftr_desc"])
train_test["norm_pos_of_features_stem_description_max2"] = map(try_divide, train_test["pos_of_features_stem_description_max2"], train_test["intersect_bigram_ftr_desc"])
train_test["norm_pos_of_features_stem_description_mean2"] = map(try_divide, train_test["pos_of_features_stem_description_mean2"], train_test["intersect_bigram_ftr_desc"])
train_test["norm_pos_of_features_stem_description_median2"] = map(try_divide, train_test["pos_of_features_stem_description_max2"], train_test["intersect_bigram_ftr_desc"])
train_test["norm_pos_of_features_stem_description_std2"] = map(try_divide, train_test["pos_of_features_stem_description_std2"] , train_test["intersect_bigram_ftr_desc"])

train_test["pos_of_features_stem_description_min3"] = map(np.min, pos)
train_test["pos_of_features_stem_description_mean3"] = map(np.mean, pos)
train_test["pos_of_features_stem_description_med3"] = map(np.median, pos)
train_test["pos_of_features_stem_description_max3"] = map(np.max, pos)
train_test["pos_of_features_stem_description_std3"] = map(np.std, pos)

train_test["norm_pos_of_features_stem_description_min3"] = map(try_divide, train_test["pos_of_features_stem_description_min3"], train_test["intersect_trigram_ftr_desc"])
train_test["norm_pos_of_features_stem_description_max3"] = map(try_divide, train_test["pos_of_features_stem_description_max3"], train_test["intersect_trigram_ftr_desc"])
train_test["norm_pos_of_features_stem_description_mean3"] = map(try_divide, train_test["pos_of_features_stem_description_mean3"], train_test["intersect_trigram_ftr_desc"])
train_test["norm_pos_of_features_stem_description_median3"] = map(try_divide, train_test["pos_of_features_stem_description_max3"], train_test["intersect_trigram_ftr_desc"])
train_test["norm_pos_of_features_stem_description_std3"] = map(try_divide, train_test["pos_of_features_stem_description_std3"] , train_test["intersect_trigram_ftr_desc"])

# extract basic distance feat : Features and Description stemmed
def JaccardCoef(A, B):
    A, B = set(A), set(B)
    intersect = len(A.intersection(B))
    union = len(A.union(B))
    coef = try_divide(intersect, union)
    return coef
def DiceDist(A, B):
    A, B = set(A), set(B)
    intersect = len(A.intersection(B))
    union = len(A) + len(B)
    d = try_divide(2*intersect, union)
    return d
def compute_dist(A, B, dist="jaccard_coef"):
    if dist == "jaccard_coef":
        d = JaccardCoef(A, B)
    elif dist == "dice_dist":
        d = DiceDist(A, B)
    return d

train_test["jac_ftr_desc_uni"] = list(train_test.apply(lambda x: compute_dist(x['features_stem'].split(' '), x['description'].split(' '), "jaccard_coef"), axis=1))
train_test["dic_ftr_desc_uni"] = list(train_test.apply(lambda x: compute_dist(x['features_stem'].split(' '), x['description'].split(' '), "dice_dist"), axis=1))
train_test["jac_ftr_desc_bi"] = list(train_test.apply(lambda x: compute_dist(x['bigram_feature'], x['bigram_description'], "jaccard_coef"), axis=1))
train_test["dic_ftr_desc_bi"] = list(train_test.apply(lambda x: compute_dist(x['bigram_feature'], x['bigram_description'], "dice_dist"), axis=1))
train_test["jac_ftr_desc_tri"] = list(train_test.apply(lambda x: compute_dist(x['trigram_feature'], x['trigram_description'], "jaccard_coef"), axis=1))
train_test["dic_ftr_desc_tri"] = list(train_test.apply(lambda x: compute_dist(x['trigram_feature'], x['trigram_description'], "dice_dist"), axis=1))

train_test.drop(['bigram_description','trigram_description','frgram_description','bigram_feature','trigram_feature','description_orig','features'],axis=1,inplace=True)


with open("../train_test0.pkl", "wb") as f:
    cPickle.dump((train_test,train_y,ntrain), f, -1)

with open("../train_test0.pkl", "rb") as f:
    (train_test,train_y,ntrain) = cPickle.load(f)

######Mgr Id
managers_count = train_test['manager_id'].value_counts()

train_test['top_1_manager'] = train_test['manager_id'].apply(lambda x: 1 if x in managers_count.index.values[
    managers_count.values >= np.percentile(managers_count.values, 99)] else 0)
train_test['top_2_manager'] = train_test['manager_id'].apply(lambda x: 1 if x in managers_count.index.values[
    managers_count.values >= np.percentile(managers_count.values, 98)] else 0)
train_test['top_5_manager'] = train_test['manager_id'].apply(lambda x: 1 if x in managers_count.index.values[
    managers_count.values >= np.percentile(managers_count.values, 95)] else 0)
train_test['top_10_manager'] = train_test['manager_id'].apply(lambda x: 1 if x in managers_count.index.values[
    managers_count.values >= np.percentile(managers_count.values, 90)] else 0)
train_test['top_15_manager'] = train_test['manager_id'].apply(lambda x: 1 if x in managers_count.index.values[
    managers_count.values >= np.percentile(managers_count.values, 85)] else 0)
train_test['top_20_manager'] = train_test['manager_id'].apply(lambda x: 1 if x in managers_count.index.values[
    managers_count.values >= np.percentile(managers_count.values, 80)] else 0)
train_test['top_25_manager'] = train_test['manager_id'].apply(lambda x: 1 if x in managers_count.index.values[
    managers_count.values >= np.percentile(managers_count.values, 75)] else 0)
train_test['top_30_manager'] = train_test['manager_id'].apply(lambda x: 1 if x in managers_count.index.values[
    managers_count.values >= np.percentile(managers_count.values, 70)] else 0)
train_test['top_50_manager'] = train_test['manager_id'].apply(lambda x: 1 if x in managers_count.index.values[
    managers_count.values >= np.percentile(managers_count.values, 50)] else 0)
train_test['bottom_10_manager'] = train_test['manager_id'].apply(lambda x: 1 if x in managers_count.index.values[
    managers_count.values >= np.percentile(managers_count.values, 10)] else 0)
train_test['bottom_20_manager'] = train_test['manager_id'].apply(lambda x: 1 if x in managers_count.index.values[
    managers_count.values >= np.percentile(managers_count.values, 20)] else 0)
train_test['bottom_30_manager'] = train_test['manager_id'].apply(lambda x: 1 if x in managers_count.index.values[
    managers_count.values >= np.percentile(managers_count.values, 30)] else 0)

###Building rating
buildings_count = train_test['building_id'].value_counts()

train_test['top_1_building'] = train_test['building_id'].apply(lambda x: 1 if x in buildings_count.index.values[
    buildings_count.values >= np.percentile(buildings_count.values, 99)] else 0)
train_test['top_2_building'] = train_test['building_id'].apply(lambda x: 1 if x in buildings_count.index.values[
    buildings_count.values >= np.percentile(buildings_count.values, 98)] else 0)
train_test['top_5_building'] = train_test['building_id'].apply(lambda x: 1 if x in buildings_count.index.values[
    buildings_count.values >= np.percentile(buildings_count.values, 95)] else 0)
train_test['top_10_building'] = train_test['building_id'].apply(lambda x: 1 if x in buildings_count.index.values[
    buildings_count.values >= np.percentile(buildings_count.values, 90)] else 0)
train_test['top_15_building'] = train_test['building_id'].apply(lambda x: 1 if x in buildings_count.index.values[
    buildings_count.values >= np.percentile(buildings_count.values, 85)] else 0)
train_test['top_20_building'] = train_test['building_id'].apply(lambda x: 1 if x in buildings_count.index.values[
    buildings_count.values >= np.percentile(buildings_count.values, 80)] else 0)
train_test['top_25_building'] = train_test['building_id'].apply(lambda x: 1 if x in buildings_count.index.values[
    buildings_count.values >= np.percentile(buildings_count.values, 75)] else 0)
train_test['top_30_building'] = train_test['building_id'].apply(lambda x: 1 if x in buildings_count.index.values[
    buildings_count.values >= np.percentile(buildings_count.values, 70)] else 0)
train_test['top_50_building'] = train_test['building_id'].apply(lambda x: 1 if x in buildings_count.index.values[
    buildings_count.values >= np.percentile(buildings_count.values, 50)] else 0)
train_test['bottom_10_building'] = train_test['building_id'].apply(lambda x: 1 if x in buildings_count.index.values[
    buildings_count.values >= np.percentile(buildings_count.values, 10)] else 0)
train_test['bottom_20_building'] = train_test['building_id'].apply(lambda x: 1 if x in buildings_count.index.values[
    buildings_count.values >= np.percentile(buildings_count.values, 20)] else 0)
train_test['bottom_30_building'] = train_test['building_id'].apply(lambda x: 1 if x in buildings_count.index.values[
    buildings_count.values >= np.percentile(buildings_count.values, 30)] else 0)

######Display address
add_count = train_test['display_address'].value_counts()

train_test['top_1_add'] = train_test['display_address'].apply(lambda x: 1 if x in add_count.index.values[
    add_count.values >= np.percentile(add_count.values, 99)] else 0)
train_test['top_2_add'] = train_test['display_address'].apply(lambda x: 1 if x in add_count.index.values[
    add_count.values >= np.percentile(add_count.values, 98)] else 0)
train_test['top_5_add'] = train_test['display_address'].apply(lambda x: 1 if x in add_count.index.values[
    add_count.values >= np.percentile(add_count.values, 95)] else 0)
train_test['top_10_add'] = train_test['display_address'].apply(lambda x: 1 if x in add_count.index.values[
    add_count.values >= np.percentile(add_count.values, 90)] else 0)
train_test['top_15_add'] = train_test['display_address'].apply(lambda x: 1 if x in add_count.index.values[
    add_count.values >= np.percentile(add_count.values, 85)] else 0)
train_test['top_20_add'] = train_test['display_address'].apply(lambda x: 1 if x in add_count.index.values[
    add_count.values >= np.percentile(add_count.values, 80)] else 0)
train_test['top_25_add'] = train_test['display_address'].apply(lambda x: 1 if x in add_count.index.values[
    add_count.values >= np.percentile(add_count.values, 75)] else 0)
train_test['top_30_add'] = train_test['display_address'].apply(lambda x: 1 if x in add_count.index.values[
    add_count.values >= np.percentile(add_count.values, 70)] else 0)
train_test['top_50_add'] = train_test['display_address'].apply(lambda x: 1 if x in add_count.index.values[
    add_count.values >= np.percentile(add_count.values, 50)] else 0)
train_test['bottom_10_add'] = train_test['display_address'].apply(lambda x: 1 if x in add_count.index.values[
    add_count.values >= np.percentile(add_count.values, 10)] else 0)
train_test['bottom_20_add'] = train_test['display_address'].apply(lambda x: 1 if x in add_count.index.values[
    add_count.values >= np.percentile(add_count.values, 20)] else 0)
train_test['bottom_30_add'] = train_test['display_address'].apply(lambda x: 1 if x in add_count.index.values[
    add_count.values >= np.percentile(add_count.values, 30)] else 0)

train_test['display_address'] = train_test['display_address'].apply(lambda x: x.lower())

address_map = {
    'w': 'west',
    'st.': 'street',
    'ave': 'avenue',
    'st': 'street',
    'e': 'east',
    'n': 'north',
    's': 'south'
}
def address_map_func(s):
    s = s.split(' ')
    out = []
    for x in s:
        if x in address_map:
            out.append(re.sub(r'[^a-zA-Z0-9\s]', '', address_map[x]).strip().lower())
        else:
            out.append(x)
    return ' '.join(out).lower()

train_test['display_address'] = train_test['display_address'].apply(lambda x: address_map_func(x))

new_cols = ['street', 'avenue', 'east', 'west', 'north', 'south']

for col in new_cols:
    train_test[col] = train_test['display_address'].apply(lambda x: 1 if col in x else 0)

train_test['other_address'] = train_test[new_cols].apply(lambda x: 1 if x.sum() == 0 else 0, axis=1)

categorical = ["display_address", "manager_id", "building_id", "street_address","listing_id"]
for f in categorical:
        if train_test[f].dtype=='object':
            #print(f)
            lbl = preprocessing.LabelEncoder()
            lbl.fit(list(train_test[f].values))
            train_test[f] = lbl.transform(list(train_test[f].values))


# save before OHC and other trransform            
with open("../train_test.pkl", "wb") as f:
    cPickle.dump((train_test,train_y,ntrain), f, -1)

#Label Encoding
train_test['bathrooms_cat'], labels = pd.factorize(train_test['bathrooms'].values, sort=True)
train_test['bedroom_cat'], labels = pd.factorize(train_test['bedrooms'].values, sort=True)
train_test['lat_cat'], labels = pd.factorize(train_test['latitude'].values, sort=True)
train_test['lon_cat'], labels = pd.factorize(train_test['longitude'].values, sort=True)
categorical.extend(['bathrooms_cat','bedroom_cat','lat_cat','lon_cat'])
            
#######Generate the actual data frames

train_df = train_test.iloc[:ntrain, :]
test_df = train_test.iloc[ntrain:, :]


features_to_use=[
#u'bathrooms', u'bedrooms', u'description', u'latitude', u'longitude',u'photos', u'price',
#'features_0','features_1', 'features_2', 'features_3', 'features_stem',
"listing_id",u'display_address',u'manager_id', u'building_id',  u'street_address', 
u'created','created_year', 'created_month', 'created_day','created_hour', 'created_weekday', 'created_wd', 
'bc_price',
'lg_price', 
'Zero_building_id', 'Zero_Ftr', 'Zero_description',
'num_description_words', 'ratio_description_words', 'num_photos','num_features', 'num_in_desc', 'num_in_ftr', 'desc_letters_count',
'bigram_cnt','trigram_cnt', 'frgram_cnt', 'uni_unigram_cnt', 'uni_bigram_cnt',
'uni_trigram_cnt', 'uni_frgram_cnt', 'rat_unigram_cnt',
'rat_bigram_cnt', 'rat_trigram_cnt', 'rat_frgram_cnt',
'intersect_ftr_desc', 'intersect_bigram_ftr_desc',
'intersect_trigram_ftr_desc',
'pos_of_features_stem_description_min',
'pos_of_features_stem_description_mean',
'pos_of_features_stem_description_med',
'pos_of_features_stem_description_max',
'pos_of_features_stem_description_std',
'norm_pos_of_features_stem_description_min',
'norm_pos_of_features_stem_description_max',
'norm_pos_of_features_stem_description_mean',
'norm_pos_of_features_stem_description_median',
'norm_pos_of_features_stem_description_std',
'pos_of_features_stem_description_min2',
'pos_of_features_stem_description_mean2',
'pos_of_features_stem_description_med2',
'pos_of_features_stem_description_max2',
'pos_of_features_stem_description_std2',
'norm_pos_of_features_stem_description_min2',
'norm_pos_of_features_stem_description_max2',
'norm_pos_of_features_stem_description_mean2',
'norm_pos_of_features_stem_description_median2',
'norm_pos_of_features_stem_description_std2',
'pos_of_features_stem_description_min3',
'pos_of_features_stem_description_mean3',
'pos_of_features_stem_description_med3',
'pos_of_features_stem_description_max3',
'pos_of_features_stem_description_std3',
'norm_pos_of_features_stem_description_min3',
'norm_pos_of_features_stem_description_max3',
'norm_pos_of_features_stem_description_mean3',
'norm_pos_of_features_stem_description_median3',
'norm_pos_of_features_stem_description_std3', 
'jac_ftr_desc_uni','dic_ftr_desc_uni', 'jac_ftr_desc_bi', 'dic_ftr_desc_bi','jac_ftr_desc_tri', 'dic_ftr_desc_tri', 
'top_1_manager','top_2_manager', 'top_5_manager', 'top_10_manager','top_15_manager', 'top_20_manager', 'top_25_manager','top_30_manager', 'top_50_manager', 'bottom_10_manager','bottom_20_manager', 'bottom_30_manager', 
'top_1_building','top_2_building', 'top_5_building', 'top_10_building','top_15_building', 'top_20_building', 'top_25_building','top_30_building', 'top_50_building', 'bottom_10_building','bottom_20_building', 'bottom_30_building', 
'top_1_add','top_2_add', 'top_5_add', 'top_10_add', 'top_15_add', 'top_20_add','top_25_add', 'top_30_add', 'top_50_add', 'bottom_10_add','bottom_20_add', 'bottom_30_add', 
'street', 'avenue', 'east','west', 'north', 'south', 'other_address', 
'bathrooms_cat','bedroom_cat', 'lat_cat', 'lon_cat']

 
with open("../pickle00.pkl", "wb") as f:
    cPickle.dump((train_df,test_df,train_y,features_to_use,ntrain,test_df_listing_id), f, -1)

with open("../pickle00.pkl", "rb") as f:
    (train_df,test_df,train_y,features_to_use,ntrain,test_df_listing_id) = cPickle.load( f)

    
train_test = pd.concat((train_df, test_df), axis=0).reset_index(drop=True)
####New features
train_test['bed_bath'] = train_test['bedroom_cat'].astype(str)+'_'+train_test['bathrooms_cat'].astype(str)
train_test['bed_bath'], labels = pd.factorize(train_test['bed_bath'].values, sort=True)

def try_divide(x, y, val=0.0):
    """ 
    	Try to divide two numbers
    """
    if y != 0.0:
    	val = float(x) / y
    else:
        val =x
    return round(val,2)

train_test['per_bed_price'] = map(try_divide,train_test['lg_price'],train_test['bedrooms'])
train_test['per_bath_price'] = map(try_divide,train_test['lg_price'],train_test['bathrooms'])
mn1 = round(train_test['per_bed_price'].mean(),2)
mn2 = round(train_test['per_bath_price'].mean(),2)
train_test['per_bed_price_dev'] = train_test['per_bed_price'] - mn1
train_test['per_bath_price_dev'] =train_test['per_bath_price'] -mn2
#boxcox-variant
train_test['per_bed_price_bc'] = map(try_divide,train_test['bc_price'],train_test['bedrooms'])
train_test['per_bath_price_bc'] = map(try_divide,train_test['bc_price'],train_test['bathrooms'])
mn1 = round(train_test['per_bed_price_bc'].mean(),2)
mn2 = round(train_test['per_bath_price_bc'].mean(),2)
train_test['per_bed_price_dev_bc'] = train_test['per_bed_price_bc'] - mn1
train_test['per_bath_price_dev_bc'] =train_test['per_bath_price_bc'] -mn2


# rounding - variant
train_test['lat_cat_rnd'], labels = pd.factorize(train_test['latitude'].apply(lambda x: round(x,3)).values, sort=True)
train_test['lon_cat_rnd'], labels = pd.factorize(train_test['longitude'].apply(lambda x: round(x,3)).values, sort=True)
train_test['lg_price_rnd'] = train_test['lg_price'].apply(lambda x:round(x,3))
train_test['bc_price_rnd'] = train_test['bc_price'].apply(lambda x:round(x,3))

train_df = train_test.iloc[:ntrain, :]
test_df = train_test.iloc[ntrain:, :]

## Target encoding 

columns_target_enc = ['manager_id','building_id','street_address','display_address','created_year', 'created_month','created_day', 'created_hour', 'created_weekday', 'created_wd','street', 'avenue', 'east','west', 'north','south', 'other_address', 'bathrooms_cat', 'bedroom_cat', 'lat_cat','lon_cat','lat_cat_rnd','lon_cat_rnd','bed_bath']

for c in columns_target_enc:
    data = train_df[[c]].copy()
    data['interest_level'] = train_y
    mgr_target_matrix = np.zeros((train_y.shape[0],1))
    mgrtst_target_matrix = np.zeros((test_df.shape[0],1))

    kf = model_selection.KFold(n_splits=nfold, shuffle=True, random_state=2017)
    for dev_index, val_index in kf.split(range(train_y.shape[0])):
        dev_X, val_X = data.iloc[dev_index,:], data.iloc[val_index,:]
        datax = dev_X.groupby(c).agg([len,np.mean,np.std])
        datax.columns = ['_'.join(col).strip() for col in datax.columns.values]
        datax = datax.loc[datax.interest_level_len>5]
        datax[c+'_tgtenc'] =  datax['interest_level_mean']
        datax.drop(['interest_level_len','interest_level_mean','interest_level_std'],axis=1,inplace=True)
        print(datax.columns)
        datatst = test_df[[c]].copy()
        val_X = val_X.join(datax,on=[c], how='left').fillna(-1)
        datatst = datatst.join(datax,on=[c], how='left').fillna(-1)
        mgr_target_matrix[val_index,...] = val_X[list(set(datax.columns)-set([c]))]
        mgrtst_target_matrix += datatst[list(set(datax.columns)-set([c]))]

    mgrtst_target_matrix = mgrtst_target_matrix/nfold
    mgr_target_matrix = pd.DataFrame(mgr_target_matrix)  
    mgr_target_matrix.columns=[c+'_tgtenc'+str(x) for x in mgr_target_matrix.columns] 
    mgrtst_target_matrix = pd.DataFrame(mgrtst_target_matrix)  
    mgrtst_target_matrix.columns=[mgr_target_matrix.columns] 
    total = pd.concat([mgr_target_matrix,mgrtst_target_matrix]).reset_index(drop=True)
    train_test = pd.concat([train_test,total],axis=1)


##Count encoding
columns_target_enc = ['manager_id','building_id','street_address','display_address','created_year', 'created_month','created_day', 'created_hour', 'created_weekday', 'created_wd','street', 'avenue', 'east','west', 'north','south', 'other_address', 'bathrooms_cat', 'bedroom_cat', 'lat_cat','lon_cat','lat_cat_rnd','lon_cat_rnd','bed_bath']


for c in columns_target_enc:
    data = train_df[[c]].copy()
    data['interest_level'] = train_y
    mgr_target_matrix = np.zeros((train_y.shape[0],1))
    mgrtst_target_matrix = np.zeros((test_df.shape[0],1))

    kf = model_selection.KFold(n_splits=nfold, shuffle=True, random_state=2017)
    for dev_index, val_index in kf.split(range(train_y.shape[0])):
        dev_X, val_X = data.iloc[dev_index,:], data.iloc[val_index,:]
        datax = dev_X.groupby(c).agg([len])
        datax.columns = ['_'.join(col).strip() for col in datax.columns.values]
        datax[c+'_countenc'] =  datax['interest_level_len']
        datax.drop(['interest_level_len'],axis=1,inplace=True)
        print(datax.columns)
        datatst = test_df[[c]].copy()
        val_X = val_X.join(datax,on=[c], how='left').fillna(-1)
        datatst = datatst.join(datax,on=[c], how='left').fillna(-1)
        mgr_target_matrix[val_index,...] = val_X[list(set(datax.columns)-set([c]))]
        mgrtst_target_matrix += datatst[list(set(datax.columns)-set([c]))]

    mgrtst_target_matrix = mgrtst_target_matrix/nfold
    mgr_target_matrix = pd.DataFrame(mgr_target_matrix)  
    mgr_target_matrix.columns=[c+'_countenc'+str(x) for x in mgr_target_matrix.columns] 
    mgrtst_target_matrix = pd.DataFrame(mgrtst_target_matrix)  
    mgrtst_target_matrix.columns=[mgr_target_matrix.columns] 
    total = pd.concat([mgr_target_matrix,mgrtst_target_matrix]).reset_index(drop=True)
    train_test = pd.concat([train_test,total],axis=1)

##RANK of Count encoding
import scipy.stats as ss

columns_target_enc = ['manager_id','building_id','street_address','display_address','created_year', 'created_month','created_day', 'created_hour', 'created_weekday', 'created_wd','street', 'avenue', 'east','west', 'north','south', 'other_address', 'bathrooms_cat', 'bedroom_cat', 'lat_cat','lon_cat','lat_cat_rnd','lon_cat_rnd','bed_bath']


for c in columns_target_enc:
    data = train_df[[c]].copy()
    data['interest_level'] = train_y
    mgr_target_matrix = np.zeros((train_y.shape[0],1))
    mgrtst_target_matrix = np.zeros((test_df.shape[0],1))

    kf = model_selection.KFold(n_splits=nfold, shuffle=True, random_state=2017)
    for dev_index, val_index in kf.split(range(train_y.shape[0])):
        dev_X, val_X = data.iloc[dev_index,:], data.iloc[val_index,:]
        datax = dev_X.groupby(c).agg([len])
        datax.columns = ['_'.join(col).strip() for col in datax.columns.values]
        datax[c+'_rankenc'] =  datax['interest_level_len']
        datax[c+'_rankenc'] =  ss.rankdata(datax[c+'_rankenc'].values)
        datax.drop(['interest_level_len'],axis=1,inplace=True)
        print(datax.columns)
        datatst = test_df[[c]].copy()
        val_X = val_X.join(datax,on=[c], how='left').fillna(-1)
        datatst = datatst.join(datax,on=[c], how='left').fillna(-1)
        mgr_target_matrix[val_index,...] = val_X[list(set(datax.columns)-set([c]))]
        mgrtst_target_matrix += datatst[list(set(datax.columns)-set([c]))]

    mgrtst_target_matrix = mgrtst_target_matrix/nfold
    mgr_target_matrix = pd.DataFrame(mgr_target_matrix)  
    mgr_target_matrix.columns=[c+'_rankenc'+str(x) for x in mgr_target_matrix.columns] 
    mgrtst_target_matrix = pd.DataFrame(mgrtst_target_matrix)  
    mgrtst_target_matrix.columns=[mgr_target_matrix.columns] 
    total = pd.concat([mgr_target_matrix,mgrtst_target_matrix]).reset_index(drop=True)
    train_test = pd.concat([train_test,total],axis=1)

####sentiment analysis of description    
from nltk.sentiment.vader import SentimentIntensityAnalyzer
dictlist = []
for sentence in train_test.description:
    sid = SentimentIntensityAnalyzer()
    ss = sid.polarity_scores(sentence)
    dictlist.append(ss)

sentimentdf = pd.DataFrame(dictlist)
train_test['compound']=sentimentdf['compound'].values
train_test['neg']=sentimentdf['neg'].values
train_test['neu']=sentimentdf['neu'].values
train_test['pos']=sentimentdf['pos'].values
train_df = train_test.iloc[:ntrain, :]
test_df = train_test.iloc[ntrain:, :]


features_to_use_ln=[
#u'bathrooms', u'bedrooms', u'description', u'latitude', u'longitude',u'photos', u'price',
#'features_0','features_1', 'features_2', 'features_3', 'features_stem',
###label encoding
"listing_id",u'building_id', u'created',u'display_address',  u'manager_id', u'street_address','created_year', 'created_month','created_day', 'created_hour', 'created_weekday', 'created_wd','bed_bath','street', 'avenue', 'east', 'west', 'north','south', 'other_address', 'bathrooms_cat', 'bedroom_cat','lat_cat','lon_cat',# 'lat_cat_rnd','lon_cat_rnd',#
#'bc_price','per_bed_price_bc','per_bath_price_bc','per_bed_price_dev_bc','per_bath_price_dev_bc',#bc_price_rnd,
'lg_price','per_bed_price','per_bath_price','per_bed_price_dev','per_bath_price_dev',  #lg_price_rnd,
###target encoding
#'manager_id_tgtenc0', 'building_id_tgtenc0','street_address_tgtenc0', 'display_address_tgtenc0','created_year_tgtenc0', 'created_month_tgtenc0','created_day_tgtenc0', 'created_hour_tgtenc0','created_weekday_tgtenc0', 'created_wd_tgtenc0', 'street_tgtenc0','avenue_tgtenc0', 'east_tgtenc0', 'west_tgtenc0', 'north_tgtenc0','south_tgtenc0', 'other_address_tgtenc0','bathrooms_cat_tgtenc0','bedroom_cat_tgtenc0', 'lat_cat_tgtenc0', 'lon_cat_tgtenc0','bed_bath_tgtenc0',#'lat_cat_rnd_tgtenc0', 'lon_cat_rnd_tgtenc0', 
###count encoding
#'manager_id_countenc0', 'building_id_countenc0','street_address_countenc0', 'display_address_countenc0','created_year_countenc0', 'created_month_countenc0','created_day_countenc0', 'created_hour_countenc0','created_weekday_countenc0', 'created_wd_countenc0', 'street_countenc0','avenue_countenc0', 'east_countenc0', 'west_countenc0', 'north_countenc0','south_countenc0', 'other_address_countenc0','bathrooms_cat_countenc0','bedroom_cat_countenc0', 'lat_cat_countenc0', 'lon_cat_countenc0', 'bed_bath_countenc0'#,'lat_cat_rnd_countenc0', 'lon_cat_rnd_countenc0',
###rank count encoding
#'manager_id_rankenc0', 'building_id_rankenc0','street_address_rankenc0', 'display_address_rankenc0','created_year_rankenc0', 'created_month_rankenc0','created_day_rankenc0', 'created_hour_rankenc0','created_weekday_rankenc0', 'created_wd_rankenc0', 'street_rankenc0','avenue_rankenc0', 'east_rankenc0', 'west_rankenc0', 'north_rankenc0','south_rankenc0', 'other_address_rankenc0','bathrooms_cat_rankenc0','bedroom_cat_rankenc0', 'lat_cat_rankenc0', 'lon_cat_rankenc0', 'bed_bath_rankenc0'#,'lat_cat_rnd_rankenc0', 'lon_cat_rnd_rankenc0',
###description attr
'compound','neg','neu','pos',
###Other features
'Zero_building_id', 'Zero_Ftr', 'Zero_description',
'num_description_words', 'ratio_description_words', 'num_photos','num_features', 'num_in_desc', 'num_in_ftr', 'desc_letters_count',
'bigram_cnt','trigram_cnt', 'frgram_cnt', 'uni_unigram_cnt', 'uni_bigram_cnt',
'uni_trigram_cnt', 'uni_frgram_cnt', 'rat_unigram_cnt',
'rat_bigram_cnt', 'rat_trigram_cnt', 'rat_frgram_cnt',
'intersect_ftr_desc', 'intersect_bigram_ftr_desc',
'intersect_trigram_ftr_desc',
'pos_of_features_stem_description_min',
'pos_of_features_stem_description_mean',
'pos_of_features_stem_description_med',
'pos_of_features_stem_description_max',
'pos_of_features_stem_description_std',
'norm_pos_of_features_stem_description_min',
'norm_pos_of_features_stem_description_max',
'norm_pos_of_features_stem_description_mean',
'norm_pos_of_features_stem_description_median',
'norm_pos_of_features_stem_description_std',
'pos_of_features_stem_description_min2',
'pos_of_features_stem_description_mean2',
'pos_of_features_stem_description_med2',
'pos_of_features_stem_description_max2',
'pos_of_features_stem_description_std2',
'norm_pos_of_features_stem_description_min2',
'norm_pos_of_features_stem_description_max2',
'norm_pos_of_features_stem_description_mean2',
'norm_pos_of_features_stem_description_median2',
'norm_pos_of_features_stem_description_std2',
'pos_of_features_stem_description_min3',
'pos_of_features_stem_description_mean3',
'pos_of_features_stem_description_med3',
'pos_of_features_stem_description_max3',
'pos_of_features_stem_description_std3',
'norm_pos_of_features_stem_description_min3',
'norm_pos_of_features_stem_description_max3',
'norm_pos_of_features_stem_description_mean3',
'norm_pos_of_features_stem_description_median3',
'norm_pos_of_features_stem_description_std3', 
'jac_ftr_desc_uni','dic_ftr_desc_uni', 'jac_ftr_desc_bi', 'dic_ftr_desc_bi','jac_ftr_desc_tri', 'dic_ftr_desc_tri', 
'top_1_manager','top_2_manager', 'top_5_manager', 'top_10_manager','top_15_manager', 'top_20_manager', 'top_25_manager','top_30_manager', 'top_50_manager', 'bottom_10_manager','bottom_20_manager', 'bottom_30_manager', 
'top_1_building','top_2_building', 'top_5_building', 'top_10_building','top_15_building', 'top_20_building', 'top_25_building','top_30_building', 'top_50_building', 'bottom_10_building','bottom_20_building', 'bottom_30_building', 
'top_1_add','top_2_add', 'top_5_add', 'top_10_add', 'top_15_add', 'top_20_add','top_25_add', 'top_30_add', 'top_50_add', 'bottom_10_add','bottom_20_add', 'bottom_30_add']

with open("../pickle01.pkl", "wb") as f:
    cPickle.dump((train_df,test_df,train_y,features_to_use,features_to_use_ln,ntrain,test_df_listing_id), f, -1)

with open("../pickle01.pkl", "rb") as f:
    (train_df,test_df,train_y,features_to_use,features_to_use_ln,ntrain,test_df_listing_id) = cPickle.load( f)

train_test = pd.concat((train_df, test_df), axis=0).reset_index(drop=True)

####New Ftr based on Brendon ideas
train_test['per_bed_bath_price'] = map(try_divide,train_test['lg_price'],train_test['bedrooms']+train_test['bathrooms'])
train_test['per_bed_bath_price'] = train_test['per_bed_bath_price'] .apply(lambda x: round(x,3))
train_test['bedPerBath'] = map(try_divide,train_test['bedrooms'],train_test['bathrooms'])
train_test['bedPerBath'] = train_test['bedPerBath'].apply(lambda x: round(x,3))
train_test['bedBathDiff'] = train_test['bedrooms']-train_test['bathrooms']
train_test['bedBathSum'] = train_test['bedrooms']+train_test['bathrooms']
train_test['bedsPerc'] = map(try_divide,train_test['bedrooms'],train_test['bedrooms']+train_test['bathrooms'])
train_test['bedsPerc'] = train_test['bedsPerc'].apply(lambda x: round(x,3))


columns_target_enc = ['manager_id','building_id']

for c in columns_target_enc:
    data = train_df[[c]].copy()
    data['interest_level'] = train_y
    data['interest_level_high'] = data['interest_level'].apply(lambda x: 1 if x==2 else 0)
    data.drop('interest_level',inplace=True,axis=1)
    #    data['interest_level_medium'] = data['interest_level'].apply(lambda x: 1 if x==1 else 0)
    mgr_target_matrix = np.zeros((train_y.shape[0],1))
    mgrtst_target_matrix = np.zeros((test_df.shape[0],1))

    kf = model_selection.KFold(n_splits=nfold, shuffle=True, random_state=2017)
    for dev_index, val_index in kf.split(range(train_y.shape[0])):
        dev_X, val_X = data.iloc[dev_index,:], data.iloc[val_index,:]
        datax = dev_X.groupby(c).agg([len,np.mean])
        datax.columns = ['_'.join(col).strip() for col in datax.columns.values]
        datax = datax.loc[datax.interest_level_high_len>5]
        datax[c+'_interest_level_high'] =  datax['interest_level_high_mean']
        datax.drop(['interest_level_high_len','interest_level_high_mean'],axis=1,inplace=True)
        print(datax.columns)
        datatst = test_df[[c]].copy()
        val_X = val_X.join(datax,on=[c], how='left').fillna(round(np.random.random_sample(),2))
        datatst = datatst.join(datax,on=[c], how='left').fillna(round(np.random.random_sample(),2))
        mgr_target_matrix[val_index,...] = val_X[list(set(datax.columns)-set([c]))]
        mgrtst_target_matrix += datatst[list(set(datax.columns)-set([c]))]

    mgrtst_target_matrix = mgrtst_target_matrix/nfold
    mgr_target_matrix = pd.DataFrame(mgr_target_matrix)  
    mgr_target_matrix.columns=[c+'_interest_level_high'+str(x) for x in mgr_target_matrix.columns] 
    mgrtst_target_matrix = pd.DataFrame(mgrtst_target_matrix)  
    mgrtst_target_matrix.columns=[mgr_target_matrix.columns] 
    total = pd.concat([mgr_target_matrix,mgrtst_target_matrix]).reset_index(drop=True)
    train_test = pd.concat([train_test,total],axis=1)

for c in columns_target_enc:
    data = train_df[[c]].copy()
    data['interest_level'] = train_y
    data['interest_level_medium'] = data['interest_level'].apply(lambda x: 1 if x==1 else 0)
    data.drop('interest_level',inplace=True,axis=1)
    mgr_target_matrix = np.zeros((train_y.shape[0],1))
    mgrtst_target_matrix = np.zeros((test_df.shape[0],1))

    kf = model_selection.KFold(n_splits=nfold, shuffle=True, random_state=2017)
    for dev_index, val_index in kf.split(range(train_y.shape[0])):
        dev_X, val_X = data.iloc[dev_index,:], data.iloc[val_index,:]
        datax = dev_X.groupby(c).agg([len,np.mean])
        datax.columns = ['_'.join(col).strip() for col in datax.columns.values]
        datax = datax.loc[datax.interest_level_medium_len>5]
        datax[c+'_interest_level_medium'] =  datax['interest_level_medium_mean']
        datax.drop(['interest_level_medium_len','interest_level_medium_mean'],axis=1,inplace=True)
        print(datax.columns)
        datatst = test_df[[c]].copy()
        val_X = val_X.join(datax,on=[c], how='left').fillna(round(np.random.random_sample(),2))
        datatst = datatst.join(datax,on=[c], how='left').fillna(round(np.random.random_sample(),2))
        mgr_target_matrix[val_index,...] = val_X[list(set(datax.columns)-set([c]))]
        mgrtst_target_matrix += datatst[list(set(datax.columns)-set([c]))]

    mgrtst_target_matrix = mgrtst_target_matrix/nfold
    mgr_target_matrix = pd.DataFrame(mgr_target_matrix)  
    mgr_target_matrix.columns=[c+'_interest_level_medium'+str(x) for x in mgr_target_matrix.columns] 
    mgrtst_target_matrix = pd.DataFrame(mgrtst_target_matrix)  
    mgrtst_target_matrix.columns=[mgr_target_matrix.columns] 
    total = pd.concat([mgr_target_matrix,mgrtst_target_matrix]).reset_index(drop=True)
    train_test = pd.concat([train_test,total],axis=1)

mn1 = round(train_test['per_bed_price'].mean(),2)
mn2 = round(train_test['per_bath_price'].mean(),2)

train_test['per_bed_price_rat'] = train_test['per_bed_price'] / mn1
train_test['per_bath_price_rat'] =train_test['per_bath_price'] / mn2

features_to_use=[
u'bathrooms', u'bedrooms', u'description', u'latitude', u'longitude',u'photos', u'price',
'features_0','features_1', 'features_2', 'features_3', 'features_stem',
###label encoding
"listing_id",u'building_id', u'created',u'display_address',  u'manager_id', u'street_address','created_year', 'created_month','created_day', 'created_hour', 'created_weekday', 'created_wd','bed_bath','street', 'avenue', 'east', 'west', 'north','south', 'other_address', 'bathrooms_cat', 'bedroom_cat','lat_cat','lon_cat', 'lat_cat_rnd','lon_cat_rnd',#
'bc_price','per_bed_price_bc','per_bath_price_bc','per_bed_price_dev_bc','per_bath_price_dev_bc','bc_price_rnd',
'lg_price','per_bed_price','per_bath_price','per_bed_price_dev','per_bath_price_dev',  'lg_price_rnd',###target encoding
'manager_id_tgtenc0', 'building_id_tgtenc0','street_address_tgtenc0', 'display_address_tgtenc0','created_year_tgtenc0', 'created_month_tgtenc0','created_day_tgtenc0', 'created_hour_tgtenc0','created_weekday_tgtenc0', 'created_wd_tgtenc0', 'street_tgtenc0','avenue_tgtenc0', 'east_tgtenc0', 'west_tgtenc0', 'north_tgtenc0','south_tgtenc0', 'other_address_tgtenc0','bathrooms_cat_tgtenc0','bedroom_cat_tgtenc0', 'lat_cat_tgtenc0', 'lon_cat_tgtenc0','bed_bath_tgtenc0','lat_cat_rnd_tgtenc0', 'lon_cat_rnd_tgtenc0', 
###count encoding
'manager_id_countenc0', 'building_id_countenc0','street_address_countenc0', 'display_address_countenc0','created_year_countenc0', 'created_month_countenc0','created_day_countenc0', 'created_hour_countenc0','created_weekday_countenc0', 'created_wd_countenc0', 'street_countenc0','avenue_countenc0', 'east_countenc0', 'west_countenc0', 'north_countenc0','south_countenc0', 'other_address_countenc0','bathrooms_cat_countenc0','bedroom_cat_countenc0', 'lat_cat_countenc0', 'lon_cat_countenc0', 'bed_bath_countenc0','lat_cat_rnd_countenc0', 'lon_cat_rnd_countenc0',
###rank count encoding
'manager_id_rankenc0', 'building_id_rankenc0','street_address_rankenc0', 'display_address_rankenc0','created_year_rankenc0', 'created_month_rankenc0','created_day_rankenc0', 'created_hour_rankenc0','created_weekday_rankenc0', 'created_wd_rankenc0', 'street_rankenc0','avenue_rankenc0', 'east_rankenc0', 'west_rankenc0', 'north_rankenc0','south_rankenc0', 'other_address_rankenc0','bathrooms_cat_rankenc0','bedroom_cat_rankenc0', 'lat_cat_rankenc0', 'lon_cat_rankenc0', 'bed_bath_rankenc0','lat_cat_rnd_rankenc0', 'lon_cat_rnd_rankenc0',
###description attr
'compound','neg','neu','pos',
###Other features
'Zero_building_id', 'Zero_Ftr', 'Zero_description',
'num_description_words', 'ratio_description_words', 'num_photos','num_features', 'num_in_desc', 'num_in_ftr', 'desc_letters_count',
'bigram_cnt','trigram_cnt', 'frgram_cnt', 'uni_unigram_cnt', 'uni_bigram_cnt',
'uni_trigram_cnt', 'uni_frgram_cnt', 'rat_unigram_cnt',
'rat_bigram_cnt', 'rat_trigram_cnt', 'rat_frgram_cnt',
'intersect_ftr_desc', 'intersect_bigram_ftr_desc',
'intersect_trigram_ftr_desc',
'pos_of_features_stem_description_min',
'pos_of_features_stem_description_mean',
'pos_of_features_stem_description_med',
'pos_of_features_stem_description_max',
'pos_of_features_stem_description_std',
'norm_pos_of_features_stem_description_min',
'norm_pos_of_features_stem_description_max',
'norm_pos_of_features_stem_description_mean',
'norm_pos_of_features_stem_description_median',
'norm_pos_of_features_stem_description_std',
'pos_of_features_stem_description_min2',
'pos_of_features_stem_description_mean2',
'pos_of_features_stem_description_med2',
'pos_of_features_stem_description_max2',
'pos_of_features_stem_description_std2',
'norm_pos_of_features_stem_description_min2',
'norm_pos_of_features_stem_description_max2',
'norm_pos_of_features_stem_description_mean2',
'norm_pos_of_features_stem_description_median2',
'norm_pos_of_features_stem_description_std2',
'pos_of_features_stem_description_min3',
'pos_of_features_stem_description_mean3',
'pos_of_features_stem_description_med3',
'pos_of_features_stem_description_max3',
'pos_of_features_stem_description_std3',
'norm_pos_of_features_stem_description_min3',
'norm_pos_of_features_stem_description_max3',
'norm_pos_of_features_stem_description_mean3',
'norm_pos_of_features_stem_description_median3',
'norm_pos_of_features_stem_description_std3', 
'jac_ftr_desc_uni','dic_ftr_desc_uni', 'jac_ftr_desc_bi', 'dic_ftr_desc_bi','jac_ftr_desc_tri', 'dic_ftr_desc_tri', 
'top_1_manager','top_2_manager', 'top_5_manager', 'top_10_manager','top_15_manager', 'top_20_manager', 'top_25_manager','top_30_manager', 'top_50_manager', 'bottom_10_manager','bottom_20_manager', 'bottom_30_manager', 
'top_1_building','top_2_building', 'top_5_building', 'top_10_building','top_15_building', 'top_20_building', 'top_25_building','top_30_building', 'top_50_building', 'bottom_10_building','bottom_20_building', 'bottom_30_building', 
'top_1_add','top_2_add', 'top_5_add', 'top_10_add', 'top_15_add', 'top_20_add','top_25_add', 'top_30_add', 'top_50_add', 'bottom_10_add','bottom_20_add', 'bottom_30_add',
'per_bed_bath_price','bedPerBath','bedBathDiff','bedBathSum','bedsPerc','per_bed_price_rat','per_bath_price_rat','manager_id_interest_level_high0','building_id_interest_level_high0','manager_id_interest_level_medium0','building_id_interest_level_medium0'
]
train_df = train_test.iloc[:ntrain, :]
test_df = train_test.iloc[ntrain:, :]

with open("../pickle03.pkl", "wb") as f:
    cPickle.dump((train_df,test_df,train_y,features_to_use,features_to_use,ntrain,test_df_listing_id), f, -1)
###########
# new features
## Faron - Cat embedding for Keras
###  Based on faron's scripts pulished for Sant Prod Prediction
import cPickle
import os
import sys
import scipy as sc
import operator
import numpy as np
import pandas as pd
from scipy import sparse
import xgboost as xgb
from sklearn import model_selection, preprocessing, ensemble
from sklearn.metrics import log_loss
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from bs4 import BeautifulSoup
#reload(sys)
#sys.setdefaultencoding('utf8')
#r = re.compile(r"\s")
from sklearn.preprocessing import LabelEncoder
import re
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import TruncatedSVD
import nltk
from scipy.stats import boxcox
from sklearn.decomposition import TruncatedSVD
import datetime as dt
from nltk.stem.porter import *
import gc
import math
from collections import Counter
    
with open("../pickle03.pkl", "rb") as f:
    (train_df,test_df,train_y,features_to_use,features_to_use,ntrain,test_df_listing_id) = cPickle.load( f)

train_test = pd.concat((train_df, test_df), axis=0).reset_index(drop=True)



num_ftrs = ['Zero_building_id', 'Zero_Ftr','Zero_description', 'num_description_words','ratio_description_words', 'num_photos', 'num_features', 'top_1_manager', 'top_2_manager','top_5_manager', 'top_10_manager', 'top_15_manager','top_20_manager', 'top_25_manager', 'top_30_manager','top_50_manager', 'bottom_10_manager', 'bottom_20_manager','bottom_30_manager', 'top_1_building', 'top_2_building','top_5_building', 'top_10_building', 'top_15_building','top_20_building', 'top_25_building', 'top_30_building','top_50_building', 'bottom_10_building', 'bottom_20_building','bottom_30_building', 'top_1_add', 'top_2_add', 'top_5_add','top_10_add', 'top_15_add', 'top_20_add', 'top_25_add','top_30_add', 'top_50_add', 'bottom_10_add', 'bottom_20_add','bottom_30_add','lg_price','per_bed_price','per_bath_price','per_bed_price_dev','per_bath_price_dev', #'lg_price_rnd',
'per_bed_bath_price','bedPerBath','bedBathDiff','bedBathSum','bedsPerc','per_bed_price_rat','per_bath_price_rat','manager_id_interest_level_high0','building_id_interest_level_high0','manager_id_interest_level_medium0','building_id_interest_level_medium0'
]
cat_ftr = ["listing_id",u'building_id', u'created',u'display_address',  u'manager_id', u'street_address','created_year', 'created_month','created_day', 'created_hour', 'created_weekday', 'created_wd','bed_bath','street', 'avenue', 'east', 'west', 'north','south', 'other_address', 'bathrooms_cat', 'bedroom_cat','lat_cat','lon_cat', #'lat_cat_rnd','lon_cat_rnd'#,
]

SEED = 0
NFOLDS = 5
NTHREADS = 4

xgb_params = {
    'seed': 0,
    'colsample_bytree': 1,
    'silent': 1,
    'subsample': 1.0,
    'learning_rate': 1.0,
    'objective': 'reg:linear',
    'max_depth': 100,
    'num_parallel_tree': 1,
    'min_child_weight': 250,
    'eval_metric': 'rmse',
    'nthread': NTHREADS,
    'nrounds': 1
}

class XgbWrapper(object):
    def __init__(self, seed=0, params=None):
        self.param = params
        self.param['seed'] = seed
        self.nrounds = params.pop('nrounds', 1000)

    def train(self, x_train, y_train, x_valid=None, y_valid=None, sample_weights=None):
        dtrain = xgb.DMatrix(x_train, label=y_train)
        self.gbdt = xgb.train(self.param, dtrain, self.nrounds)
    
    # pred_leaf=True => getting leaf indices
    def predict(self, x):
        return self.gbdt.predict(xgb.DMatrix(x), pred_leaf=True).astype(int)




clf = XgbWrapper(seed=SEED, params=xgb_params)

ntrain = train_df.shape[0]
x_train = train_df[cat_ftr]
x_test = test_df[cat_ftr]
dtrain = xgb.DMatrix(x_train, label=train_y)
oof_train = np.zeros((ntrain,))

kf = model_selection.KFold(n_splits=NFOLDS, shuffle=True, random_state=SEED).split(x_train)

for i, (train_index, test_index) in enumerate(kf):
    x_tr = x_train.loc[train_index]
    y_tr = train_y[train_index]
    x_te = x_train.loc[test_index]

    clf.train(x_tr, y_tr)
    oof_train[test_index] = clf.predict((x_te))

clf.train(x_train, train_y)
oof_test = clf.predict((x_test))
oof_train=oof_train.reshape(-1, 1)
oof_test=oof_test.reshape(-1, 1)

ntrain = x_train.shape[0]

train_test = np.concatenate((oof_train, oof_test)).reshape(-1, )
min_obs = 10
# replace infrequent values by nan
val = dict((k, np.nan if v < min_obs else k) for k, v in dict(Counter(train_test)).items())
k, v = np.array(list(zip(*sorted(val.items()))))
train_test = v[np.digitize(train_test, k, right=True)]
ohe = pd.get_dummies(train_test, dummy_na=False, sparse=True)

train_test = pd.concat((train_df[num_ftrs], test_df[num_ftrs]), axis=0).reset_index(drop=True)

from sklearn.preprocessing import MinMaxScaler, LabelEncoder
scaler = MinMaxScaler()

for col in num_ftrs:
    train_test[col] = scaler.fit_transform(train_test[col])

#Required for big sparse matrix
#from scipy.sparse import lil_matrix
#
#def sparse_df_to_array(df):
#    """ Convert sparse dataframe to sparse array csr_matrix used by
#    scikit learn. """
#    arr = lil_matrix(df.shape, dtype=np.float32)
#    for i, col in enumerate(df.columns):
#        ix = df[col] != 0
#        arr[np.where(ix), i] = df.ix[ix, col]
#
#    return arr.tocsr()
  
#ohe = sparse_df_to_array(ohe)

x_train_ohe = ohe.loc[:ntrain]
x_test_ohe = ohe.loc[ntrain:]

print("OneHotEncoded XG-Embeddings: {},{}".format(x_train_ohe.shape, x_test_ohe.shape))


train_df_catenc = pd.concat([train_df[num_ftrs],x_train_ohe],axis=1).fillna(0)
test_df_catenc = pd.concat([test_df[num_ftrs],x_test_ohe],axis=1).fillna(0)
#save
with open("../pickle02_catenc_forKeras.pkl", "wb") as f:
    cPickle.dump((train_df,test_df,train_y,train_df_catenc,test_df_catenc,features_to_use,cat_ftr,num_ftrs,ntrain,test_df_listing_id), f, -1)

##############################
## Ftr Prep for normal Keras - Scale all ftrs
import cPickle
import os
import sys
import scipy as sc
import operator
import numpy as np
import pandas as pd
from scipy import sparse
import xgboost as xgb
from sklearn import model_selection, preprocessing, ensemble
from sklearn.metrics import log_loss
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from bs4 import BeautifulSoup
#reload(sys)
#sys.setdefaultencoding('utf8')
#r = re.compile(r"\s")
from sklearn.preprocessing import LabelEncoder
import re
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import TruncatedSVD
import nltk
from scipy.stats import boxcox
from sklearn.decomposition import TruncatedSVD
import datetime as dt
from nltk.stem.porter import *
import gc
import math
from collections import Counter
    
with open("../pickle03.pkl", "rb") as f:
    (train_df,test_df,train_y,features_to_use,features_to_use,ntrain,test_df_listing_id) = cPickle.load( f)

train_test = pd.concat((train_df, test_df), axis=0).reset_index(drop=True)

from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler,  RobustScaler
scaler = MinMaxScaler()

features_to_use1=[
#u'bathrooms', u'bedrooms', u'description', u'latitude', u'longitude',u'photos', u'price',
#'features_0','features_1', 'features_2', 'features_3', 'features_stem',
###label encoding
"listing_id",u'building_id', u'created',u'display_address',  u'manager_id', u'street_address','created_year', 'created_month','created_day', 'created_hour', 'created_weekday', 'created_wd','bed_bath','street', 'avenue', 'east', 'west', 'north','south', 'other_address', 'bathrooms_cat', 'bedroom_cat','lat_cat','lon_cat', 'lat_cat_rnd','lon_cat_rnd',#
'bc_price','per_bed_price_bc','per_bath_price_bc','per_bed_price_dev_bc','per_bath_price_dev_bc','bc_price_rnd',
'lg_price','per_bed_price','per_bath_price','per_bed_price_dev','per_bath_price_dev',  'lg_price_rnd',###target encoding
'manager_id_tgtenc0', 'building_id_tgtenc0','street_address_tgtenc0', 'display_address_tgtenc0','created_year_tgtenc0', 'created_month_tgtenc0','created_day_tgtenc0', 'created_hour_tgtenc0','created_weekday_tgtenc0', 'created_wd_tgtenc0', 'street_tgtenc0','avenue_tgtenc0', 'east_tgtenc0', 'west_tgtenc0', 'north_tgtenc0','south_tgtenc0', 'other_address_tgtenc0','bathrooms_cat_tgtenc0','bedroom_cat_tgtenc0', 'lat_cat_tgtenc0', 'lon_cat_tgtenc0','bed_bath_tgtenc0','lat_cat_rnd_tgtenc0', 'lon_cat_rnd_tgtenc0', 
###count encoding
'manager_id_countenc0', 'building_id_countenc0','street_address_countenc0', 'display_address_countenc0','created_year_countenc0', 'created_month_countenc0','created_day_countenc0', 'created_hour_countenc0','created_weekday_countenc0', 'created_wd_countenc0', 'street_countenc0','avenue_countenc0', 'east_countenc0', 'west_countenc0', 'north_countenc0','south_countenc0', 'other_address_countenc0','bathrooms_cat_countenc0','bedroom_cat_countenc0', 'lat_cat_countenc0', 'lon_cat_countenc0', 'bed_bath_countenc0','lat_cat_rnd_countenc0', 'lon_cat_rnd_countenc0',
###rank count encoding
'manager_id_rankenc0', 'building_id_rankenc0','street_address_rankenc0', 'display_address_rankenc0','created_year_rankenc0', 'created_month_rankenc0','created_day_rankenc0', 'created_hour_rankenc0','created_weekday_rankenc0', 'created_wd_rankenc0', 'street_rankenc0','avenue_rankenc0', 'east_rankenc0', 'west_rankenc0', 'north_rankenc0','south_rankenc0', 'other_address_rankenc0','bathrooms_cat_rankenc0','bedroom_cat_rankenc0', 'lat_cat_rankenc0', 'lon_cat_rankenc0', 'bed_bath_rankenc0','lat_cat_rnd_rankenc0', 'lon_cat_rnd_rankenc0',
###description attr
'compound','neg','neu','pos',
###Other features
'Zero_building_id', 'Zero_Ftr', 'Zero_description',
'num_description_words', 'ratio_description_words', 'num_photos','num_features', 'num_in_desc', 'num_in_ftr', 'desc_letters_count',
'bigram_cnt','trigram_cnt', 'frgram_cnt', 'uni_unigram_cnt', 'uni_bigram_cnt',
'uni_trigram_cnt', 'uni_frgram_cnt', 'rat_unigram_cnt',
'rat_bigram_cnt', 'rat_trigram_cnt', 'rat_frgram_cnt',
'intersect_ftr_desc', 'intersect_bigram_ftr_desc',
'intersect_trigram_ftr_desc',
'pos_of_features_stem_description_min',
'pos_of_features_stem_description_mean',
'pos_of_features_stem_description_med',
'pos_of_features_stem_description_max',
'pos_of_features_stem_description_std',
'norm_pos_of_features_stem_description_min',
'norm_pos_of_features_stem_description_max',
'norm_pos_of_features_stem_description_mean',
'norm_pos_of_features_stem_description_median',
'norm_pos_of_features_stem_description_std',
'pos_of_features_stem_description_min2',
'pos_of_features_stem_description_mean2',
'pos_of_features_stem_description_med2',
'pos_of_features_stem_description_max2',
'pos_of_features_stem_description_std2',
'norm_pos_of_features_stem_description_min2',
'norm_pos_of_features_stem_description_max2',
'norm_pos_of_features_stem_description_mean2',
'norm_pos_of_features_stem_description_median2',
'norm_pos_of_features_stem_description_std2',
'pos_of_features_stem_description_min3',
'pos_of_features_stem_description_mean3',
'pos_of_features_stem_description_med3',
'pos_of_features_stem_description_max3',
'pos_of_features_stem_description_std3',
'norm_pos_of_features_stem_description_min3',
'norm_pos_of_features_stem_description_max3',
'norm_pos_of_features_stem_description_mean3',
'norm_pos_of_features_stem_description_median3',
'norm_pos_of_features_stem_description_std3', 
'jac_ftr_desc_uni','dic_ftr_desc_uni', 'jac_ftr_desc_bi', 'dic_ftr_desc_bi','jac_ftr_desc_tri', 'dic_ftr_desc_tri', 
'top_1_manager','top_2_manager', 'top_5_manager', 'top_10_manager','top_15_manager', 'top_20_manager', 'top_25_manager','top_30_manager', 'top_50_manager', 'bottom_10_manager','bottom_20_manager', 'bottom_30_manager', 
'top_1_building','top_2_building', 'top_5_building', 'top_10_building','top_15_building', 'top_20_building', 'top_25_building','top_30_building', 'top_50_building', 'bottom_10_building','bottom_20_building', 'bottom_30_building', 
'top_1_add','top_2_add', 'top_5_add', 'top_10_add', 'top_15_add', 'top_20_add','top_25_add', 'top_30_add', 'top_50_add', 'bottom_10_add','bottom_20_add', 'bottom_30_add',
'per_bed_bath_price','bedPerBath','bedBathDiff','bedBathSum','bedsPerc','per_bed_price_rat','per_bath_price_rat','manager_id_interest_level_high0','building_id_interest_level_high0','manager_id_interest_level_medium0','building_id_interest_level_medium0'
]


for col in features_to_use1:
    train_test[col] = scaler.fit_transform(train_test[col])

train_df = train_test.loc[:ntrain]
test_df = train_test.loc[ntrain:]

#save
with open("../pickle04_minmax_forKeras.pkl", "wb") as f:
    cPickle.dump((train_df,test_df,train_y,features_to_use,ntrain,test_df_listing_id), f, -1)

import cPickle
import os
import sys
import scipy as sc
import operator
import numpy as np
import pandas as pd
from scipy import sparse
import xgboost as xgb
from sklearn import model_selection, preprocessing, ensemble
from sklearn.metrics import log_loss
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from bs4 import BeautifulSoup
#reload(sys)
#sys.setdefaultencoding('utf8')
#r = re.compile(r"\s")
from sklearn.preprocessing import LabelEncoder
import re
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import TruncatedSVD
import nltk
from scipy.stats import boxcox
from sklearn.decomposition import TruncatedSVD
import datetime as dt
from nltk.stem.porter import *
import gc
import math
from collections import Counter
    
with open("../pickle03.pkl", "rb") as f:
    (train_df,test_df,train_y,features_to_use,features_to_use,ntrain,test_df_listing_id) = cPickle.load( f)

train_test = pd.concat((train_df, test_df), axis=0).reset_index(drop=True)

from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler,  RobustScaler
scaler = StandardScaler()

for col in features_to_use1:
    train_test[col] = scaler.fit_transform(train_test[col])

train_df = train_test.loc[:ntrain]
test_df = train_test.loc[ntrain:]

#save
with open("../pickle04_stdscl_forKeras.pkl", "wb") as f:
    cPickle.dump((train_df,test_df,train_y,features_to_use,ntrain,test_df_listing_id), f, -1)

import cPickle
import os
import sys
import scipy as sc
import operator
import numpy as np
import pandas as pd
from scipy import sparse
import xgboost as xgb
from sklearn import model_selection, preprocessing, ensemble
from sklearn.metrics import log_loss
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from bs4 import BeautifulSoup
#reload(sys)
#sys.setdefaultencoding('utf8')
#r = re.compile(r"\s")
from sklearn.preprocessing import LabelEncoder
import re
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import TruncatedSVD
import nltk
from scipy.stats import boxcox
from sklearn.decomposition import TruncatedSVD
import datetime as dt
from nltk.stem.porter import *
import gc
import math
from collections import Counter
    
with open("../pickle03.pkl", "rb") as f:
    (train_df,test_df,train_y,features_to_use,features_to_use,ntrain,test_df_listing_id) = cPickle.load( f)

train_test = pd.concat((train_df, test_df), axis=0).reset_index(drop=True)

from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler,  RobustScaler
scaler = RobustScaler()

for col in features_to_use1:
    train_test[col] = scaler.fit_transform(train_test[col])

train_df = train_test.loc[:ntrain]
test_df = train_test.loc[ntrain:]

#save
with open("../pickle04_robscl_forKeras.pkl", "wb") as f:
    cPickle.dump((train_df,test_df,train_y,features_to_use,ntrain,test_df_listing_id), f, -1)

###
#Some mgr id ftr
import cPickle
import os
import sys
import scipy as sc
import operator
import numpy as np
import pandas as pd
from scipy import sparse
import xgboost as xgb
from sklearn import model_selection, preprocessing, ensemble
from sklearn.metrics import log_loss
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from bs4 import BeautifulSoup
#reload(sys)
#sys.setdefaultencoding('utf8')
#r = re.compile(r"\s")
from sklearn.preprocessing import LabelEncoder
import re
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import TruncatedSVD
import nltk
from scipy.stats import boxcox
from sklearn.decomposition import TruncatedSVD
import datetime as dt
from nltk.stem.porter import *
import gc
import math
from collections import Counter
    
with open("../pickle03.pkl", "rb") as f:
    (train_df,test_df,train_y,features_to_use,features_to_use,ntrain,test_df_listing_id) = cPickle.load( f)

train_df['interest_level'] = train_y

index=list(range(train_df.shape[0]))
np.random.shuffle(index)
a=[np.nan]*len(train_df)
b=[np.nan]*len(train_df)
c=[np.nan]*len(train_df)

for i in range(5):
    building_level={}
    for j in train_df['manager_id'].values:
        building_level[j]=[0,0,0]
    test_index=index[int((i*train_df.shape[0])/5):int(((i+1)*train_df.shape[0])/5)]
    train_index=list(set(index).difference(test_index))
    for j in train_index:
        temp=train_df.iloc[j]
        if temp['interest_level']==2:
            building_level[temp['manager_id']][0]+=1
        if temp['interest_level']==1:
            building_level[temp['manager_id']][1]+=1
        if temp['interest_level']==0:
            building_level[temp['manager_id']][2]+=1
    for j in test_index:
        temp=train_df.iloc[j]
        if sum(building_level[temp['manager_id']])!=0:
            a[j]=building_level[temp['manager_id']][0]*1.0/sum(building_level[temp['manager_id']])
            b[j]=building_level[temp['manager_id']][1]*1.0/sum(building_level[temp['manager_id']])
            c[j]=building_level[temp['manager_id']][2]*1.0/sum(building_level[temp['manager_id']])
train_df['manager_level_low']=a
train_df['manager_level_medium']=b
train_df['manager_level_high']=c
def try_divide(x, y, val=0.0):
    """ 
    	Try to divide two numbers
    """
    if y != 0.0:
    	val = float(x) / y
    return val
a=[]
b=[]
c=[]
building_level={}
for j in train_df['manager_id'].values:
    building_level[j]=[0,0,0]
for j in range(train_df.shape[0]):
    temp=train_df.iloc[j]
    if temp['interest_level']==2:
        building_level[temp['manager_id']][0]+=1
    if temp['interest_level']==1:
        building_level[temp['manager_id']][1]+=1
    if temp['interest_level']==0:
        building_level[temp['manager_id']][2]+=1

for i in test_df['manager_id'].values:
    if i not in building_level.keys():
        a.append(0)
        b.append(0)
        c.append(0)
    else:
        a.append(try_divide(building_level[i][0]*1.0,sum(building_level[i])))
        b.append(try_divide(building_level[i][1]*1.0,sum(building_level[i])))
        c.append(try_divide(building_level[i][2]*1.0,sum(building_level[i])))
test_df['manager_level_low']=a
test_df['manager_level_medium']=b
test_df['manager_level_high']=c

features_to_use.append('manager_level_low') 
features_to_use.append('manager_level_medium') 
features_to_use.append('manager_level_high')
######
a=[np.nan]*len(train_df)
b=[np.nan]*len(train_df)
c=[np.nan]*len(train_df)

for i in range(5):
    building_level={}
    for j in train_df['manager_id'].values:
        building_level[j]=[0,0,0]
    test_index=index[int((i*train_df.shape[0])/5):int(((i+1)*train_df.shape[0])/5)]
    train_index=list(set(index).difference(test_index))
    for j in train_index:
        temp=train_df.iloc[j]
        if temp['interest_level']==2:
            building_level[temp['manager_id']][0]+=temp['lg_price']
        if temp['interest_level']==1:
            building_level[temp['manager_id']][1]+=temp['lg_price']
        if temp['interest_level']==0:
            building_level[temp['manager_id']][2]+=temp['lg_price']
    for j in test_index:
        temp=train_df.iloc[j]
        if sum(building_level[temp['manager_id']])!=0:
            a[j]=building_level[temp['manager_id']][0]*1.0/sum(building_level[temp['manager_id']])
            b[j]=building_level[temp['manager_id']][1]*1.0/sum(building_level[temp['manager_id']])
            c[j]=building_level[temp['manager_id']][2]*1.0/sum(building_level[temp['manager_id']])
train_df['manager_level_low_pr']=a
train_df['manager_level_medium_pr']=b
train_df['manager_level_high_pr']=c
def try_divide(x, y, val=0.0):
    """ 
    	Try to divide two numbers
    """
    if y != 0.0:
    	val = float(x) / y
    return val
a=[]
b=[]
c=[]
building_level={}
for j in train_df['manager_id'].values:
    building_level[j]=[0,0,0]
for j in range(train_df.shape[0]):
    temp=train_df.iloc[j]
    if temp['interest_level']=='low':
        building_level[temp['manager_id']][0]+=temp['lg_price']
    if temp['interest_level']=='medium':
        building_level[temp['manager_id']][1]+=temp['lg_price']
    if temp['interest_level']=='high':
        building_level[temp['manager_id']][2]+=temp['lg_price']

for i in test_df['manager_id'].values:
    if i not in building_level.keys():
        a.append(0)
        b.append(0)
        c.append(0)
    else:
        a.append(try_divide(building_level[i][0]*1.0,sum(building_level[i])))
        b.append(try_divide(building_level[i][1]*1.0,sum(building_level[i])))
        c.append(try_divide(building_level[i][2]*1.0,sum(building_level[i])))
test_df['manager_level_low_pr']=a
test_df['manager_level_medium_pr']=b
test_df['manager_level_high_pr']=c

features_to_use.append('manager_level_low_pr') 
features_to_use.append('manager_level_medium_pr') 
features_to_use.append('manager_level_high_pr')




with open("../pickle05.pkl", "wb") as f:
    cPickle.dump((train_df,test_df,train_y,features_to_use,ntrain,test_df_listing_id), f, -1)

## Ftr prep for Keras and tsne


### Standard Scaler

### Robust Scaler

### XGB on count encode +log Price

### XGB on rank count encode +log Price

### XGB on target encode + log round


### XGB on count encode +box cox price

### XGB on count encode +box cox price round
 
### XGB on count encode +box cox price

### XGB on rank count encode+box cox price

### XGB Regression on count encode +log price

### XGB Regression on label encode +log price

### XGB Regression on rank encode +boxcox price

### XGB Regression on label encode +boxcox price rnd

### XGB One vs all


#####Keras Cat embedding - multi

#####Keras Cat embedding - regression


#####Keras on XGB embedded ftr - multi

#####Keras on XGB embedded ftr - regression

#####Keras regression - rank encoding

### Vowpal

### RGF

### ET

### ET - rank encoding - regression

### KNN - 2,4,8,16,64


### FTLE, LightGBM, LIBFM


########################### 
#Feature imp
###########################
from matplotlib import pylab as plt

def create_feature_map(features):
    outfile = open('xgb.fmap', 'w')
    i = 0
    for feat in features:
        outfile.write('{0}\t{1}\tq\n'.format(i, feat))
        i = i + 1

    outfile.close()
    
create_feature_map(features_to_use_ln)
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


##To try further
##TSNE

### K-mean of lat long

## xgbfi interactions

