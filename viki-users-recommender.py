from __future__ import division
from sklearn.preprocessing import StandardScaler

import pandas as pd
import time
import csv
import numpy as np
import os
import datetime
import math
import warnings
warnings.filterwarnings("ignore")

top_videos_limit = 50
sim_features = ['sim_country', 'sim_language', 'sim_adult',
                'sim_content_owner_id', 'sim_broadcast', 'sim_episode_count',
                'sim_genres', 'sim_cast',
                'jaccard_1_3', 'jaccard_2_3', 'jaccard_3_3', 'jaccard_high', 'sim_cosine_mv_ratio']
weight_features = [3,3,5,
                   0,0,0,
                   5,5,
                   0,0,15,15,45]
weight_scores = [1,3,15]

""" weight_scores:
How important a video user watched affects his recommended videos
In order words, if A watch V1 for 5% (score = 1) of its duration, and V2 for 95% (score = 3) of its duration,
V2 will be `weight_score[2] / weight_score[0]` times more important than V1 in respect to A's recommendations
"""

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
    videos_matrix = pd.read_csv('./data/videos_similarity_matrix.csv')
    # video_id and its min date_hour
    videos = behaviors.groupby('video_id').agg({'date_hour': np.min})
    videos['video_id'] = videos.index
    videos = videos.reset_index(drop=True)
    # Remove unused columns
    behaviors = behaviors.drop(['date_hour','mv_ratio'], 1)
    return (behaviors, users, videos, videos_matrix)

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

def combined_scores(behaviors, users, videos_matrix,videos_performance):
    """ To combine all similar score of all moviews calculated based on user history.
        Result will be each user and his / her top 3 recommendations (if available)
        >>> user_combined_scores = combined_scores(behaviors,videos_matrix)
        >>> user_combined_scores[:3]
                 recommendations  user_id
        0  [TV266, TV248, TV239]        1
        1  [TV266, TV248, TV239]        3
        2  [TV248, TV239, TV234]        4
    """
    # remove videos not in `top_videos_limit`
    best_videos_m = videos_performance.sort('hotness_m', ascending=False).video_id.tolist() # list of best_videos rank by hotness overall
    best_videos_f = videos_performance.sort('hotness_f', ascending=False).video_id.tolist() # list of best_videos rank by hotness overall
    best_videos_o = videos_performance.sort('hotness_o', ascending=False).video_id.tolist() # list of best_videos rank by hotness overall
    videos_matrix = videos_matrix[[x in best_videos_m[:top_videos_limit] + \
                                        best_videos_f[:top_videos_limit] + \
                                        best_videos_o[:top_videos_limit] \
                                   for x in videos_matrix['video_id_right']]]
    # Remove duplicates  (left = right)
    videos_matrix = videos_matrix[videos_matrix['video_id_left'] != videos_matrix['video_id_right']]
    # Feature scaling:
    scaler = StandardScaler(with_mean=False,with_std=True)
    for col in sim_features:
        scaler.fit(list(videos_matrix[col]))
        videos_matrix[col] = scaler.transform(videos_matrix[col])
    def sim_combined(row):
        score = 0
        for i in range(0, len(sim_features)):
            score += row[sim_features[i]]*weight_features[i]
        return score
    videos_matrix['sim_combined'] = videos_matrix.apply(sim_combined, axis=1)
    # only take those with score > 0
    videos_matrix = videos_matrix[videos_matrix['sim_combined'] > 0]
    # Top 5 similar videos to each video - 5 should be enough since we are only recommending 3 videos per person
    videos_matrix = videos_matrix.sort(['sim_combined'], ascending=False).groupby('video_id_left').head(5)
    videos_matrix = videos_matrix.drop(['sim_country', 'sim_language', 'sim_adult', 'sim_content_owner_id',
                                        'sim_broadcast', 'sim_season', 'sim_episode_count', 'sim_genres',
                                        'sim_cast', 'jaccard_1_3', 'jaccard_2_3', 'jaccard_3_3',
                                        'jaccard_high', 'sim_cosine_mv_ratio'],1)
    if videos_matrix.empty:
        user_history_videos_matrix = behaviors.reindex_axis(behaviors.columns.union(videos_matrix.columns), axis=1)
    else:
        user_history_videos_matrix = pd.merge(behaviors, videos_matrix, left_on=['video_id'], right_on=['video_id_left'])
    def weighted_sim_combined(row): # to combine with each session score
        return weight_scores[row['score']-1] * row['sim_combined']
    user_history_videos_matrix['weighted_sim_combined'] = user_history_videos_matrix.apply(weighted_sim_combined, axis=1)
    user_history_videos_matrix = user_history_videos_matrix.drop(['score','video_id_left','sim_combined'], 1)
    user_combined_scores = user_history_videos_matrix.groupby(
        ['user_id', 'video_id_right'],as_index=False).agg({'weighted_sim_combined' : np.sum})
    # filter out videos user have watched
    behaviors = behaviors.drop('score', 1)
    grouped_behaviors = pd.DataFrame({ 'video_ids' : behaviors.groupby('user_id').apply(lambda x: list(x.video_id))}) # user_id, list_of_video_ids
    grouped_behaviors['user_id'] = grouped_behaviors.index
    user_combined_scores = pd.merge(user_combined_scores, grouped_behaviors, on=['user_id'])
    try:
        user_combined_scores = user_combined_scores[user_combined_scores.apply(lambda x: x['video_id_right'] not in x['video_ids'], axis=1)]
    except: # empty dataframe
        pass
    user_combined_scores = user_combined_scores.drop('video_ids', 1) # user_id, video_id_right, weighted_sim_combined
    # produce result: user - top 3 videos (one entry per user)
    user_combined_scores = pd.merge(user_combined_scores, videos_performance, left_on=['video_id_right'], right_on='video_id', how='left').drop('video_id',1)
    user_combined_scores = pd.merge(user_combined_scores, users, on='user_id', how='left')
    def weighted_sim_combined(row): # to combine with hotness and freshness per gender
        #TODO: parameterize this
        try:
            if row['gender'] == 'm':
                return row['weighted_sim_combined'] * row['hotness_m'] * math.pow(row['freshness'],2)
            elif row['gender'] == 'f':
                return row['weighted_sim_combined'] * row['hotness_f'] * math.pow(row['freshness'],2)
            else:
                return row['weighted_sim_combined'] * row['hotness_o'] * math.pow(row['freshness'],2)
        except:
            return 0
    user_combined_scores['weighted_sim_combined'] = user_combined_scores.apply(weighted_sim_combined, axis=1)
    # # Only take the top 70 percentile of CF best scorers by excluding those that has score less than 0.15*mean
    # user_combined_scores = user_combined_scores[user_combined_scores.weighted_sim_combined > 0.15*user_combined_scores.weighted_sim_combined.mean()]
    user_combined_scores = user_combined_scores.sort(['weighted_sim_combined'], ascending=False).groupby('user_id').head(3)
    try:
        user_combined_scores = pd.DataFrame({ 'recommendations' : user_combined_scores.groupby('user_id').apply(lambda x: list(x.video_id_right))})
        user_combined_scores['user_id'] = user_combined_scores.index
    except: # empty dataframe
        user_combined_scores=user_combined_scores.drop(['video_id_right','country','gender',
                                                        'hotness_m','hotness_f','hotness_o','freshness',
                                                        'bestness_m','bestness_f','bestness_o'],1)
        user_combined_scores.columns = ['recommendations','user_id']
    return user_combined_scores.reset_index(drop=True)

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
    print str(datetime.datetime.now()) + " => Processing data... "
    (behaviors, users, videos, videos_matrix) = read_data()
    print str(datetime.datetime.now()) + " => Calculating videos' hotness and freshness..."
    videos_performance = compute_videos_performance(behaviors, users, videos)
    print str(datetime.datetime.now()) + " => Combining results for each user..."
    user_combined_scores = combined_scores(behaviors,users,videos_matrix,videos_performance)
    print str(datetime.datetime.now()) + " => Processing recommendations..."
    (submit1,submit2) = processing_recommendations(user_combined_scores,behaviors,users,videos)
    print str(datetime.datetime.now()) + " => Output to csv..."
    output_result_to_csv(submit1,submit2)

if __name__ == "__main__":
    main()