from __future__ import division
from sklearn.preprocessing import StandardScaler

import pandas as pd
import time
import csv
import numpy as np
import os
import datetime
import math

# below is for SVD
# Bash needed
# rm first line
# tail -n +2 "data/20150701094451-Behavior_training.csv" > data/behavior.csv
# ml conversion
# awk -F',' '{print $2","$3","$5}' data/behavior.csv > data/behavior-ml-score.csv # User / TV / Score
# awk -F',' '{print $2","$3","$4}' data/behavior.csv > data/behavior-ml-ratio.csv # User / TV / Score

# http://ocelma.net/software/python-recsys/build/html/quickstart.html (check last part)
# http://tedlab.mit.edu/~dr/SVDLIBC/
# http://tedlab.mit.edu/~dr/SVDLIBC/svdlibc.tgz
# make
# cp bin/svd /usr/local/bin/svd

#https://github.com/ocelma/python-recsys
import recsys.algorithm
from recsys.algorithm.factorize import SVD
from recsys.utils.svdlibc import SVDLIBC

recsys.algorithm.VERBOSE = True

def process_svd(preload):
    if preload:
        svd = SVD(filename='./data/svd-all') # Loading already computed SVD model
    else:
        print "Reading data..."
        svdlibc = SVDLIBC('./data/behavior-ml-score.csv')
        svdlibc.to_sparse_matrix(sep=',', format={'col':0, 'row':1, 'value':2, 'ids': str})
        k=100
        print "Computing SVD..."
        svdlibc.compute(k)
        svd = svdlibc.export()
        svd.save_model('./data/svd-all', options={'k': k})
    #svd.predict('TV268', 9, 1, 3)
    return svd

def read_data():
    """ Read and pre-process data
        >>> (behaviors, users, videos, videos_matrix) = read_data()
        >>> behaviors[:2]
            user_id   video_id   score
        0    759744      TV003       1
        1    759744      TV015       2
        >>> users[:2]
           user_id     country gender
        0        1  Country001   None
        1        2  Country002   None
        >>> videos[:2]
               date_hour video_id
        0  2014-10-01T00    TV001
        1  2014-10-09T16    TV002
        >>> videos_matrix[:2]
               video_id_left video_id_right  sim_combined
        110109         TV177          TV462     97.387301
        287206         TV462          TV004     97.387301
    """
    users = pd.read_csv('./data/20150701094451-User_attributes.csv')
    behaviors = pd.read_csv('./data/20150701094451-Behavior_training.csv')
    # video_id and its min date_hour
    videos = behaviors.groupby('video_id').agg({'date_hour': np.min})
    videos['video_id'] = videos.index
    videos = videos.reset_index(drop=True)
    # Remove unused columns
    behaviors = behaviors.drop(['date_hour','mv_ratio'], 1)
    return (behaviors, users, videos)

def compute_videos_performance(behaviors, users, videos):
    """ To compute a list of hot videos specific to each user, removing those that he/she already watched
        >>> videos_performance = compute_videos_performance(behaviors, users, videos)
        >>> videos_performance[:5]
          video_id  hotness_m  hotness_f  freshness   hotness_o
        0    TV001   2.508197  15.295082   0.008197   23.721311
        1    TV002   0.000000   0.008772   0.008772    0.008772
        2    TV003   6.844262  61.926230   0.008197  104.245902
        3    TV004   0.000000   0.016807   0.008403    0.042017
        4    TV005   2.737705  20.811475   0.008197   35.827869
    """
    behaviors = pd.merge(behaviors, users, on='user_id', how='left')
    videos_views_high_m = behaviors[behaviors['score']> 1][behaviors['gender']=='m'].groupby('video_id').agg(['count'])
    videos_views_high_f = behaviors[behaviors['score']> 1][behaviors['gender']=='f'].groupby('video_id').agg(['count'])
    videos_views_high_o = behaviors[behaviors['score']> 1][behaviors['gender']!='f'][behaviors['gender']!='m'].groupby('video_id').agg(['count'])
    def hotness_m(row):
        try:
            first_date = datetime.datetime.strptime(row['date_hour'],"%Y-%m-%dT%H").date()
            last_date = datetime.datetime.strptime('2015-01-31', "%Y-%m-%d").date()
            user_watched = videos_views_high_m[videos_views_high_m.index==row['video_id']].reset_index().score['count'][0]
            return  user_watched / (last_date-first_date).days
        except:
            return 0
    videos['hotness_m'] = videos.apply(hotness_m, axis=1)
    def hotness_f(row):
        try:
            first_date = datetime.datetime.strptime(row['date_hour'],"%Y-%m-%dT%H").date()
            last_date = datetime.datetime.strptime('2015-01-31', "%Y-%m-%d").date()
            user_watched = videos_views_high_f[videos_views_high_f.index==row['video_id']].reset_index().score['count'][0]
            return  user_watched / (last_date-first_date).days
        except:
            return 0
    videos['hotness_f'] = videos.apply(hotness_f, axis=1)
    def hotness_o(row):
        try:
            first_date = datetime.datetime.strptime(row['date_hour'],"%Y-%m-%dT%H").date()
            last_date = datetime.datetime.strptime('2015-01-31', "%Y-%m-%d").date()
            user_watched = videos_views_high_o[videos_views_high_o.index==row['video_id']].reset_index().score['count'][0]
            return  user_watched / (last_date-first_date).days
        except:
            return 0
    videos['hotness_o'] = videos.apply(hotness_o, axis=1)
    def freshness(row):
        try:
            first_date = datetime.datetime.strptime(row['date_hour'],"%Y-%m-%dT%H").date()
            last_date = datetime.datetime.strptime('2015-01-31', "%Y-%m-%d").date()
            return  1 / (last_date-first_date).days
        except:
            return 0
    videos['freshness'] = videos.apply(freshness, axis=1)
    return videos.drop('date_hour',1)

def processing_recommendations(user_combined_scores,behaviors,users,videos):
    # processing list of unwatched hot videos for each user
    hot_videos_m = videos.sort('hotness_m', ascending=False).video_id.tolist()
    hot_videos_f = videos.sort('hotness_f', ascending=False).video_id.tolist()
    hot_videos_o = videos.sort('hotness_o', ascending=False).video_id.tolist()
    users_history = behaviors.groupby('user_id',as_index=False).agg(lambda x: ' '.join(x.video_id)).drop('score', 1)
    users_history = pd.merge(users_history,users, on='user_id', how='right')
    def hot_videos_unwatched(row):
        if row['gender'] == 'm':
            hot_videos = hot_videos_m
        elif row['gender'] == 'f':
            hot_videos = hot_videos_f
        else:
            hot_videos = hot_videos_o
        try:
            watched = set([item for item in row['video_id'].split()])
            return [x  for x in hot_videos if x not in watched]
        except: # never watched anything
            return hot_videos
    users_history['hot_videos_unwatched'] = users_history.apply(hot_videos_unwatched, axis=1)
    users_hot_videos = users_history.drop('video_id',1)
    # separated by '-1,DEXTRA' and '-2,DEXTRA' (removed, otherwise we can't use `row['count'] % 3` below)
    test1 = pd.read_csv('./data/20150701094451-Sample_submission-p1.csv')
    test2 = pd.read_csv('./data/20150701094451-Sample_submission-p2.csv')
    # merge to preserve ordering of the test sets
    submit1 = pd.merge(test1, user_combined_scores, on=['user_id'], how='left')
    submit1 = pd.merge(submit1, users_hot_videos, on=['user_id'], how='left')
    submit1['count'] = submit1.index
    submit2 = pd.merge(test2, user_combined_scores, on=['user_id'], how='left')
    submit2 = pd.merge(submit2, users_hot_videos, on=['user_id'], how='left')
    submit2['count'] = submit2.index
    def merge(r1,r2): # merging 2 list, removing those from r2 that already appears on r1 - with priority on list #1
        return r1 +  [x for x in r2 if x not in r1]
    def recommendation(row):
        try:
            if len(row['recommendations']) == 3: # If we have enough recommendations
                rec = row['recommendations'][row['count'] % 3]
            else: # welp, not enough
                rec = merge(row['recommendations'],row['hot_videos_unwatched'])[row['count'] % 3]
        except: # row['recommendations'] could be NaN
            try: # see if there is any `hot_videos_unwatched` available for the user
                rec = row['hot_videos_unwatched'][row['count'] % 3]
            except: # when user haven't watched anything - no record in behaviors file.
                rec = hot_videos[row['count'] % 3]
        return rec
    submit1['video_id'] = submit1.apply(recommendation, axis=1)
    submit2['video_id'] = submit2.apply(recommendation, axis=1)
    submit1 = submit1.drop(['recommendations','count','hot_videos_unwatched','country','gender'], 1)
    submit2 = submit2.drop(['recommendations','count','hot_videos_unwatched','country','gender'], 1)
    return (submit1,submit2)

def output_result_to_csv(submit1,submit2):
    if not os.path.exists('result/'):
        os.makedirs('result/')
    with open('./result/submit-'+'-'.join(str(x) for x in weight_features)+
              '-'+'-'.join(str(x) for x in weight_scores)+'-top-'+
              str(top_videos_limit)+'.csv', 'w') as f:
        writer = csv.writer(f, lineterminator='\n')
        writer.writerow(submit1.columns)
        writer.writerows(submit1.values)
        writer.writerow(['-1','DEXTRA'])
        writer.writerows(submit2.values)
        writer.writerow(['-2','DEXTRA'])

def main():
    print str(datetime.datetime.now()) + " => Processing SVD... "
    svd = process_svd(preload=True)
    print str(datetime.datetime.now()) + " => Processing data... "
    (behaviors, users, videos) = read_data()
    print str(datetime.datetime.now()) + " => Calculating videos' hotness and freshness..."
    videos_performance = compute_videos_performance(behaviors, users, videos)
    print str(datetime.datetime.now()) + " => Combining results for each user..."
    user_combined_scores = combined_scores(behaviors,users,videos_matrix,videos_performance)
    print str(datetime.datetime.now()) + " => Processing recommendations..."
    (submit1,submit2) = processing_recommendations(user_combined_scores,behaviors,users,videos)
    print str(datetime.datetime.now()) + " => Output to csv..."
    output_result_to_csv(submit1,submit2)
