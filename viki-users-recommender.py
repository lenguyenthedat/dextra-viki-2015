from __future__ import division
from sklearn.preprocessing import MinMaxScaler

import pandas as pd
import time
import csv
import numpy as np
import os
import datetime
import re

# Remove pandas warning
pd.options.mode.chained_assignment = None

## ==================== Data preparation
print "=> Reading data"
print datetime.datetime.now()
behaviors = pd.read_csv('./data/20150701094451-Behavior_training.csv')
test = pd.read_csv('./data/20150701094451-Sample_submission.csv')
videos_matrix = pd.read_csv('./data/videos_similarity_matrix.csv')

# Hot videos
# Top 10, should be enough # remember to exclude from watched_videos later on
# TODO: Remove similar from the top3 list. Make sure not recommending the same
# Not too urgently - it's very rare that this happen, because someone has to have less than 3 recommendations.
hot_videos = behaviors.groupby('video_id').agg(['count']).sort([('score', 'count')], ascending=False).head(10).index.tolist()
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
sim_features = ['sim_gender', 'sim_country', 'sim_language',
                'sim_adult', 'sim_content_owner_id', 'sim_broadcast',
                'sim_episode_count', 'sim_genres', 'sim_cast',
                'jaccard_1', 'jaccard_2', 'jaccard_3']
weight_features = [0,0,0,
                   0,0,0,
                   0,0,0,
                   1,1,1]
scaler = MinMaxScaler()
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

previous_user_id = -1
previous_user_id_count = 0
previous_user_id_top3 = []
for index, row in test.iterrows():
  if (index % 100 == 0 & index != 0):
    print "Finished 100 recommendations"
    print datetime.datetime.now()
  current_user_id = row['user_id']
  # print "Recommending for user#" + str(current_user_id) + " ..."
  if current_user_id > 0: # otherwise just continue - to follow dextra / viki submission format
    # work out the rank of the video we need to recommend (#1, #2, or #3)
    if current_user_id == previous_user_id: # already calculated
      previous_user_id_count += 1
      top3 = previous_user_id_top3
    else:
      previous_user_id = current_user_id
      previous_user_id_count = 1
      # re-calculating top3
      # user_history to join with videos_matrix
      user_history = behaviors[behaviors['user_id'] == current_user_id]
      user_history_videos_matrix = pd.merge(user_history, videos_matrix, left_on=['video_id'], right_on=['video_id_left'])
      # remove videos user already watched out of recommendations
      criterion = user_history_videos_matrix['video_id_right'].map(lambda x: x not in user_history['video_id'].tolist())
      user_history_videos_matrix = user_history_videos_matrix[criterion]
      # combined to get best videos and their scores
      if user_history_videos_matrix.empty: # top videos
        top3 = hot_videos
      else: #personalized
        user_history_videos_matrix = user_history_videos_matrix.groupby(['user_id', 'video_id_right'],as_index=False).aggregate(np.sum)
        # sort result
        top3 = user_history_videos_matrix.sort(['sim_combined'], ascending=False).head(3)['video_id_right'].tolist()
      previous_user_id_top3 = top3
  # assign recommendation depending on ranking
  row['video_id'] = top3[previous_user_id_count-1]
  # print "-> Video ID#" + row['video_id']
if not os.path.exists('result/'): os.makedirs('result/')
test.to_csv('./result/submit-'+'-'.join(str(x) for x in weight_features)+'.csv', encoding='utf-8', index=False)
