from __future__ import division
from sklearn.preprocessing import StandardScaler

import pandas as pd
import time
import csv
import numpy as np
import os
import datetime
import math

sim_features = ['sim_country', 'sim_language', 'sim_adult',
                'sim_content_owner_id', 'sim_broadcast', 'sim_episode_count',
                'sim_genres', 'sim_cast', 'hotness',
                'jaccard_1_3', 'jaccard_2_3', 'jaccard_3_3']
weight_features = [0,0,0,
                   0,0,0,
                   0,0,15,
                   50,50,50]
weight_scores = [1,2,3] # How important a video user watched affects his recommended videos
                        # In order words, if A watch V1 for 5% (score = 1) of its duration, and V2 for 95% (score = 3) of its duration,
                        # V2 will be `weight_score[2] / weight_score[0]` times more important than V1 in respect to A's recommendations

## ==================== Data preparation
print "=> Reading data & Pre Processing"
print datetime.datetime.now()

# Behavior
behaviors = pd.read_csv('./data/20150701094451-Behavior_training.csv')
behaviors = behaviors.drop('date_hour', 1)
behaviors = behaviors.drop('mv_ratio', 1)

# Hot videos
videos_views = behaviors.groupby('video_id').agg(['count']).sort([('score', 'count')], ascending=False)
videos = pd.read_csv('./data/20150701094451-Video_attributes.csv')

def hotness(row): # How hot the RIGHT video is, regardless of the left one
    try:
        bf_date = datetime.datetime.strptime(row['broadcast_from'], "%Y-%m").date()
        day_2015_02 = datetime.datetime.strptime('2015-02', "%Y-%m").date()
        user_watched = videos_views[videos_views.index==row['video_id']].reset_index().score['count'][0]
        return  user_watched / (day_2015_02-bf_date).days
        # return  user_watched / math.pow((day_2015_02-bf_date).days,2)
    except:
        return 0

videos['hotness'] = videos.apply(hotness, axis=1)

hot_videos = videos.sort('hotness', ascending=False).video_id.tolist()
# 50 Insignificant Videos that should be removed from the master videos_matrix
not_hot_videos = behaviors.groupby('video_id').agg(['count']).sort([('score', 'count')], ascending=False).tail(50).index.tolist()

# Videos Matrix
videos_matrix = pd.read_csv('./data/videos_similarity_matrix.csv')
# remove self-similarity entries
# It's important to do this so that we will not get skewed result - bad for our scaler
videos_matrix = videos_matrix[videos_matrix['video_id_left'] != videos_matrix['video_id_right']]
videos_matrix = videos_matrix[[x not in not_hot_videos for x in videos_matrix['video_id_left']]]
videos_matrix = videos_matrix[[x not in not_hot_videos for x in videos_matrix['video_id_right']]]

# Feature scaling:
print "=> Feature scaling"
print datetime.datetime.now()
scaler = StandardScaler()
for col in sim_features:
    scaler.fit(list(videos_matrix[col]))
    videos_matrix[col] = scaler.transform(videos_matrix[col])

# Combined
print "=> Combined weighted feature similarity"
print datetime.datetime.now()
def sim_combined(row):
    score = 0
    for i in range(0, len(sim_features)):
        if sim_features[i] > 0: # with standard scaler, feature similarity can be below zero.
                                # do not punish them in this case, esp. with jaccard scores.
          score += row[sim_features[i]]*weight_features[i]
    return score

videos_matrix['sim_combined'] = videos_matrix.apply(sim_combined, axis=1)
# Top 5 similar videos to each video - 5 should be enough since we are only recommending 3 videos per person
videos_matrix = videos_matrix.sort(['sim_combined'], ascending=False).groupby('video_id_left').head(5)

print "=> Combining matrixes"
print datetime.datetime.now()
user_history_videos_matrix = pd.merge(behaviors, videos_matrix, left_on=['video_id'], right_on=['video_id_left'])

def weighted_sim_combined(row):
    return weight_scores[row['score']-1] * row['sim_combined']

user_history_videos_matrix['weighted_sim_combined'] = user_history_videos_matrix.apply(weighted_sim_combined, axis=1)
user_history_videos_matrix = user_history_videos_matrix.drop('score', 1)
user_history_videos_matrix = user_history_videos_matrix.drop('video_id_left', 1)
user_history_videos_matrix = user_history_videos_matrix.drop('sim_combined', 1)

# group on user_level - video_id
print "=> Combining matrixes - user video level"
print datetime.datetime.now()
grouped_user_history_videos_matrix = user_history_videos_matrix.groupby(['user_id', 'video_id_right'],as_index=False).aggregate(np.sum)

# For filtering purpose (i.e do not recommend watched videos)
behaviors = behaviors.drop('score', 1)
grouped_behaviors = pd.DataFrame({ 'video_ids' : behaviors.groupby('user_id').apply(lambda x: list(x.video_id))}) # user_id, list_of_video_ids
grouped_behaviors['user_id'] = grouped_behaviors.index
grouped_user_history_videos_matrix = pd.merge(grouped_user_history_videos_matrix, grouped_behaviors, on=['user_id'])

grouped_user_history_videos_matrix = grouped_user_history_videos_matrix[grouped_user_history_videos_matrix.apply(lambda x: x['video_id_right'] not in x['video_ids'], axis=1)]
grouped_user_history_videos_matrix = grouped_user_history_videos_matrix.drop('video_ids', 1) # user_id, video_id_right, weighted_sim_combined

# Excluding blacklisted country
# we have: customer - customer country - recommended video - recommended video countries

# user - top3 videos (on user level now)
grouped_user_history_videos_matrix = grouped_user_history_videos_matrix.sort(['weighted_sim_combined'], ascending=False).groupby('user_id').head(3)
grouped_user_history_videos_matrix = pd.DataFrame({ 'recommendations' : grouped_user_history_videos_matrix.groupby('user_id').apply(lambda x: list(x.video_id_right))})
grouped_user_history_videos_matrix['user_id'] = grouped_user_history_videos_matrix.index

print "=> Processing results"
print datetime.datetime.now()
# separated by '-1,DEXTRA' and '-2,DEXTRA' (removed, otherwise we can't use `row['count'] % 3` below)
test1 = pd.read_csv('./data/20150701094451-Sample_submission-p1.csv')
test2 = pd.read_csv('./data/20150701094451-Sample_submission-p2.csv')
submit1 = pd.merge(test1, grouped_user_history_videos_matrix, on=['user_id'], how='left')
submit1['count'] = submit1.index
submit2 = pd.merge(test2, grouped_user_history_videos_matrix, on=['user_id'], how='left')
submit2['count'] = submit2.index
def recommendation(row):
  try:
    return row['recommendations'][row['count'] % 3]
  except:
    return hot_videos[row['count'] % 3]

submit1['video_id'] = submit1.apply(recommendation, axis=1)
submit2['video_id'] = submit2.apply(recommendation, axis=1)
submit1 = submit1.drop('recommendations', 1)
submit1 = submit1.drop('count', 1)
submit2 = submit2.drop('recommendations', 1)
submit2 = submit2.drop('count', 1)

print "=> Writing result to CSV"
print datetime.datetime.now()
if not os.path.exists('result/'):
    os.makedirs('result/')
with open('./result/submit-'+'-'.join(str(x) for x in weight_features)+'-'+'-'.join(str(x) for x in weight_scores)+'.csv', 'w') as f:
    writer = csv.writer(f, lineterminator='\n')
    writer.writerow(submit1.columns)
    writer.writerows(submit1.values)
    writer.writerow(['-1','DEXTRA'])
    writer.writerows(submit2.values)
    writer.writerow(['-2','DEXTRA'])
