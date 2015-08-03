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

# Users - hot videos
users_videos = behaviors.groupby('user_id',as_index=False).agg(lambda x: ' '.join(x.video_id)).drop('score', 1)

print datetime.datetime.now()
def hot_videos_unwatched(row): # people who do not like LEFT but like RIGHT
    try:
        watched = set([item for item in row['video_id'].split()])
        return [x  for x in hot_videos if x not in watched]
    except: # never watched anything
        return hot_videos

users_videos['hot_videos_unwatched'] = users_videos.apply(hot_videos_unwatched, axis=1)
users_videos = users_videos.drop('video_id',1)


print "=> Processing results"
print datetime.datetime.now()
# separated by '-1,DEXTRA' and '-2,DEXTRA' (removed, otherwise we can't use `row['count'] % 3` below)
test1 = pd.read_csv('./data/20150701094451-Sample_submission-p1.csv')
test2 = pd.read_csv('./data/20150701094451-Sample_submission-p2.csv')
submit1 = pd.merge(test1, users_videos, on=['user_id'], how='left')
submit1['count'] = submit1.index
submit2 = pd.merge(test2, users_videos, on=['user_id'], how='left')
submit2['count'] = submit2.index
def recommendation(row):
  try:
    return row['hot_videos_unwatched'][row['count'] % 3]
  except:
    return hot_videos[row['count'] % 3]

submit1['video_id'] = submit1.apply(recommendation, axis=1)
submit2['video_id'] = submit2.apply(recommendation, axis=1)
submit1 = submit1.drop('hot_videos_unwatched', 1)
submit1 = submit1.drop('count', 1)
submit2 = submit2.drop('hot_videos_unwatched', 1)
submit2 = submit2.drop('count', 1)

print "=> Writing result to CSV"
print datetime.datetime.now()
if not os.path.exists('result/'):
    os.makedirs('result/')

with open('./result/submit.csv', 'w') as f:
    writer = csv.writer(f, lineterminator='\n')
    writer.writerow(submit1.columns)
    writer.writerows(submit1.values)
    writer.writerow(['-1','DEXTRA'])
    writer.writerows(submit2.values)
    writer.writerow(['-2','DEXTRA'])
