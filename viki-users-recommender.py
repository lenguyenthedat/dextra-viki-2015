from __future__ import division
from sklearn.preprocessing import StandardScaler

import pandas as pd
import time
import csv
import numpy as np
import os
import datetime
import re

## ==================== Data preparation
print "=> Reading data"
print datetime.datetime.now()
videos_matrix = pd.read_csv('./Data/videos_similarity_matrix.csv',sep='\t')
## ===================== Combined
# Feature scaling:
print "=> Feature scaling"
print datetime.datetime.now()
sim_features = ['sim_country','sim_language', 'sim_adult', 'sim_content_owner_id', 'sim_broadcast', 'sim_episode_count', 'sim_genres', 'sim_cast', 'jaccard']
weight_features = [10,10,5,1,1,1,2,2,50]
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

behaviors = pd.read_csv('./Data/20150701094451-Behavior_training.csv')
test = pd.read_csv('./Data/20150701094451-Sample_submission.csv')

hot_videos = behaviors.groupby('video_id').agg(['count']).sort([('date_hour', 'count')], ascending=False).head(3).index.tolist()

previous_user_id = -1
previous_user_id_count = 0
previous_user_id_top3 = []
for index, row in test.iterrows():
  if (index % 100000 == 0 & index != 0):
    print "Finished 100000 recommendations"
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
test.to_csv('./result/submit-'+'-'.join(str(x) for x in weight_features)+'.csv', sep=',', encoding='utf-8', index=False)
