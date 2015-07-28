from __future__ import division
from sklearn.preprocessing import StandardScaler

import pandas as pd
import time
import csv
import numpy as np
import os
import datetime
import re

# Remove pandas warning
pd.options.mode.chained_assignment = None

sim_features = ['sim_gender','sim_country','sim_language', 'sim_adult', 'sim_content_owner_id', 'sim_broadcast', 'sim_episode_count', 'sim_genres', 'sim_cast', 'jaccard']
weight_features = [10,5,5,5,1,1,1,5,5,50]
## ==================== Data preparation
print "=> Reading data & Pre Processing"
print datetime.datetime.now()

# Hot videos
# Top 10, should be enough # remember to exclude from watched_videos later on
# TODO: Remove similar from the top3 list. Make sure not recommending the same
# Not too urgently - it's very rare that this happen, because someone has to have less than 3 recommendations.
hot_videos_f = pd.read_csv('./data/hot_videos_f.csv', sep='\t', encoding='utf-8')['video_id'].tolist()
hot_videos_m = pd.read_csv('./data/hot_videos_m.csv', sep='\t', encoding='utf-8')['video_id'].tolist()

# Videos Matrix
videos_matrix = pd.read_csv('./Data/videos_similarity_matrix.csv',sep='\t')
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
        score += row[sim_features[i]]*weight_features[i]
    return score

videos_matrix['sim_combined'] = videos_matrix.apply(sim_combined, axis=1)
videos_matrix = videos_matrix.drop('sim_country', 1)
videos_matrix = videos_matrix.drop('sim_language', 1)
videos_matrix = videos_matrix.drop('sim_adult', 1)
videos_matrix = videos_matrix.drop('sim_content_owner_id', 1)
videos_matrix = videos_matrix.drop('sim_broadcast', 1)
videos_matrix = videos_matrix.drop('sim_season', 1)
videos_matrix = videos_matrix.drop('sim_episode_count', 1)
videos_matrix = videos_matrix.drop('sim_genres', 1)
videos_matrix = videos_matrix.drop('sim_cast', 1)
videos_matrix = videos_matrix.drop('jaccard', 1)
# remove self-similarity entries
videos_matrix = videos_matrix[videos_matrix['video_id_left'] != videos_matrix['video_id_right']]
# Top 5 similar videos to each video - 5 should be enough since we are only recommending 3 videos per person
videos_matrix = videos_matrix.sort(['sim_combined'], ascending=False).groupby('video_id_left').head(5)

# Behavior
behaviors = pd.read_csv('./Data/20150701094451-Behavior_training.csv')
behaviors = behaviors.drop('date_hour', 1)
behaviors = behaviors.drop('mv_ratio', 1)

print "=> Combining matrixes"
print datetime.datetime.now()
user_history_videos_matrix = pd.merge(behaviors, videos_matrix, left_on=['video_id'], right_on=['video_id_left'])

# weighted_sim_combined
def weighted_sim_combined(row):
    return row['score'] * row['sim_combined']

user_history_videos_matrix['weighted_sim_combined'] = user_history_videos_matrix.apply(weighted_sim_combined, axis=1)
user_history_videos_matrix = user_history_videos_matrix.drop('score', 1)
user_history_videos_matrix = user_history_videos_matrix.drop('video_id_left', 1)
user_history_videos_matrix = user_history_videos_matrix.drop('sim_combined', 1)

# group on user_level - video_id
grouped_user_history_videos_matrix = user_history_videos_matrix.groupby(['user_id', 'video_id_right'],as_index=False).aggregate(np.sum)

# For filtering purpose (i.e do not recommend watched videos)
behaviors = behaviors.drop('score', 1)
grouped_behaviors = pd.DataFrame({ 'video_ids' : behaviors.groupby('user_id').apply(lambda x: list(x.video_id))}) # user_id, list_of_video_ids
grouped_behaviors['user_id'] = grouped_behaviors.index
grouped_user_history_videos_matrix = pd.merge(grouped_user_history_videos_matrix, grouped_behaviors, on=['user_id'])

grouped_user_history_videos_matrix = grouped_user_history_videos_matrix[grouped_user_history_videos_matrix.apply(lambda x: x['video_id_right'] not in x['video_ids'], axis=1)]
grouped_user_history_videos_matrix = grouped_user_history_videos_matrix.drop('video_ids', 1) # user_id, video_id_right, weighted_sim_combined

# user - top3 videos (on user level now)
grouped_user_history_videos_matrix = grouped_user_history_videos_matrix.sort(['weighted_sim_combined'], ascending=False).groupby('user_id').head(3)
grouped_user_history_videos_matrix = pd.DataFrame({ 'recommendations' : grouped_user_history_videos_matrix.groupby('user_id').apply(lambda x: list(x.video_id_right))})
grouped_user_history_videos_matrix['user_id'] = grouped_user_history_videos_matrix.index

print "=> Processing results"
print datetime.datetime.now()
# separated by '-1,DEXTRA' and '-2,DEXTRA' (removed, otherwise we can't use `row['count'] % 3` below)
test1 = pd.read_csv('./Data/20150701094451-Sample_submission-p1.csv')
test2 = pd.read_csv('./Data/20150701094451-Sample_submission-p2.csv')
users = pd.read_csv('./Data/20150701094451-User_attributes.csv')
submit1 = pd.merge(test1, grouped_user_history_videos_matrix, on=['user_id'], how='left')
submit1 = pd.merge(submit1, users, on=['user_id'], how='left')
submit1['count'] = submit1.index
submit2 = pd.merge(test2, grouped_user_history_videos_matrix, on=['user_id'], how='left')
submit2 = pd.merge(submit2, users, on=['user_id'], how='left')
submit2['count'] = submit2.index
def recommendation(row):
  try:
    return row['recommendations'][row['count'] % 3]
  except:
    if row['gender'] == 'm':
        return hot_videos_m[row['count'] % 3]
    else:
        return hot_videos_f[row['count'] % 3]

submit1['video_id'] = submit1.apply(recommendation, axis=1)
submit2['video_id'] = submit2.apply(recommendation, axis=1)
submit1 = submit1.drop('recommendations', 1)
submit1 = submit1.drop('count', 1)
submit1 = submit1.drop('country', 1)
submit1 = submit1.drop('gender', 1)
submit2 = submit2.drop('recommendations', 1)
submit2 = submit2.drop('count', 1)
submit2 = submit2.drop('country', 1)
submit2 = submit2.drop('gender', 1)

print "=> Writing result to CSV"
print datetime.datetime.now()
if not os.path.exists('result/'):
    os.makedirs('result/')
with open('./result/submit-'+'-'.join(str(x) for x in weight_features)+'.csv', 'w') as f:
    writer = csv.writer(f, lineterminator='\n')
    writer.writerow(submit1.columns)
    writer.writerows(submit1.values)
    writer.writerow(['-1','DEXTRA'])
    writer.writerows(submit2.values)
    writer.writerow(['-2','DEXTRA'])
