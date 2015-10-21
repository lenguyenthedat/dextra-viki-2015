# Bash needed
# rm first line
# tail -n +2 "data/20150701094451-Behavior_training.csv" > data/behavior.csv
# ml conversion
# awk -F',' '{gsub("TV", "", $3) ; gsub("-", "", $1) ; gsub("T", "", $1) ; print $2"::"$3"::"$5}' data/behavior.csv > data/behavior-ml.csv

#!/usr/bin/python
"""MovieLense Tutorial
see doc/tutorial.rst for more information.
"""

import numpy as np
from rsvd import MovieLensDataset
dataset = MovieLensDataset.loadDat('./data/behavior-ml.csv')
ratings=dataset.ratings()

# make sure that the ratings a properly shuffled
np.random.shuffle(ratings)

# create train, validation and test sets.
n = int(ratings.shape[0]*0.8)
train = ratings[:n]
test = ratings[n:]
v = int(train.shape[0]*0.9)
val = train[v:]
train = train[:v]

from rsvd import RSVD
dims = (dataset.movieIDs().shape[0], dataset.userIDs().shape[0])
model = RSVD.train(20, train, dims, probeArray=val,
                   learnRate=0.0005, regularization=0.005)

sqerr=0.0
for movieID,userID,rating in test:
    err = rating - model(movieID,userID)
    sqerr += err * err
sqerr /= test.shape[0]
print "Test RMSE: ", np.sqrt(sqerr)


##########
from recsys.algorithm.factorize import SVD
svd = SVD()
svd.load_data(filename='./data/behavior-ml.csv',
            sep='::',
            format={'col':0, 'row':1, 'value':2, 'ids': int})

k = 100
svd.compute(k=k,
            min_values=10,
            pre_normalize=None,
            mean_center=True,
            post_normalize=True,
            savefile='/tmp/movielens')

ITEMID1 = 1    # Toy Story (1995)
ITEMID2 = 2355 # A bug's life (1998)

svd.similarity(ITEMID1, ITEMID2)
# 0.67706936677315799

svd.similar(ITEMID1)

# Returns: <ITEMID, Cosine Similarity Value>
[(1,    0.99999999999999978), # Toy Story
 (3114, 0.87060391051018071), # Toy Story 2
 (2355, 0.67706936677315799), # A bug's life
 (588,  0.5807351496754426),  # Aladdin
 (595,  0.46031829709743477), # Beauty and the Beast
 (1907, 0.44589398718134365), # Mulan
 (364,  0.42908159895574161), # The Lion King
 (2081, 0.42566581277820803), # The Little Mermaid
 (3396, 0.42474056361935913), # The Muppet Movie
 (2761, 0.40439361857585354)] # The Iron Giant

 MIN_RATING = 0.0
MAX_RATING = 5.0
ITEMID = 1
USERID = 1

svd.predict(ITEMID, USERID, MIN_RATING, MAX_RATING)
# Predicted value 5.0

svd.get_matrix().value(ITEMID, USERID)
# Real value 5.0

svd.recommend(USERID, is_row=False) #cols are users and rows are items, thus we set is_row=False

# Returns: <ITEMID, Predicted Rating>
[(2905, 5.2133848204673416), # Shaggy D.A., The
 (318,  5.2052108435956033), # Shawshank Redemption, The
 (2019, 5.1037438278755474), # Seven Samurai (The Magnificent Seven)
 (1178, 5.0962756861447023), # Paths of Glory (1957)
 (904,  5.0771405690055724), # Rear Window (1954)
 (1250, 5.0744156653222436), # Bridge on the River Kwai, The
 (858,  5.0650911066862907), # Godfather, The
 (922,  5.0605327279819408), # Sunset Blvd.
 (1198, 5.0554543765500419), # Raiders of the Lost Ark
 (1148, 5.0548789542105332)] # Wrong Trousers, The

 svd.recommend(ITEMID)

# Returns: <USERID, Predicted Rating>
[(283,  5.716264440514446),
 (3604, 5.6471765418323141),
 (5056, 5.6218800339214496),
 (446,  5.5707524860615738),
 (3902, 5.5494529168484652),
 (4634, 5.51643364021289),
 (3324, 5.5138903299082802),
 (4801, 5.4947999354188548),
 (1131, 5.4941438045650068),
 (2339, 5.4916048051511659)]

 ####

from __future__ import division
from sklearn.preprocessing import StandardScaler

import pandas as pd
import time
import csv
import numpy as np
import os
import datetime
import math
behaviors = pd.read_csv('./data/20150701094451-Behavior_training.csv')
# user - video matrix
behaviors_wide = pd.pivot_table(behaviors, values=["score"],
                         index=["video_id", "user_id"],
                         aggfunc=np.mean).unstack()

# any cells that are missing data (i.e. a user didn't buy a particular product)
# we're going to set to 0
behaviors_wide = behaviors_wide.fillna(0)

import numpy as np

U, sigma, V = np.linalg.svd(behaviors_wide)
print "V = "
print np.round(V, decimals=2)


# this is the key. we're going to use cosine_similarity from scikit-learn
# to compute the distance between all beers
print "calculating similarity"
cosine_video_matrix = cosine_similarity(behaviors_wide)

# stuff the distance matrix into a dataframe so it's easier to operate on
cosine_video_matrix = pd.DataFrame(cosine_video_matrix, columns=behaviors_wide.index)

# give the indicies (equivalent to rownames in R) the name of the product id
cosine_video_matrix.index = cosine_video_matrix.columns

def sim_cosine_score(row):
    try:
        return cosine_video_matrix[row['video_id_left']][row['video_id_right']]
    except: # no data for row['video_id_left']
        return 0

videos_matrix['sim_cosine_score'] = videos_matrix.apply(sim_cosine_score, axis=1)


#######
import pandas as pd
import numpy as np
from scipy.sparse.linalg import svds
import scipy.sparse as sp
import pickle
data_file = pd.read_table(r'./data/behavior-ml.csv', sep = '::', header=None)

users = np.unique(data_file[0])
movies = np.unique(data_file[1])

number_of_rows = len(users)
number_of_columns = len(movies)

movie_indices, user_indices = {}, {}

for i in range(len(movies)):
    movie_indices[movies[i]] = i

for i in range(len(users)):
    user_indices[users[i]] = i


#scipy sparse matrix to store the 1M matrix
V = sp.lil_matrix((number_of_rows, number_of_columns))

#adds data into the sparse matrix
for line in data_file.values:
    u, i , r  = map(int,line)
    V[user_indices[u], movie_indices[i]] = r

#as these operations consume a lot of time, it's better to save processed data
with open('./data/viki.pickle', 'wb') as handle:
    pickle.dump(V, handle)

#as these operations consume a lot of time, it's better to save processed data
#gets SVD components from 10M matrix
u,s, vt = svds(V, k = 500)

with open('./data/viki_svd_u.pickle', 'wb') as handle:
    pickle.dump(u, handle)
with open('./data/viki_svd_s.pickle', 'wb') as handle:
    pickle.dump(s, handle)
with open('./data/viki_svd_vt.pickle', 'wb') as handle:
    pickle.dump(vt, handle)

s_diag_matrix = np.zeros((s.shape[0], s.shape[0]))

for i in range(s.shape[0]):
    s_diag_matrix[i,i] = s[i]

X_lr = np.dot(np.dot(u, s_diag_matrix), vt)