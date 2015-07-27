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
videos = pd.read_csv('./Data/20150701094451-Video_attributes.csv')
casts = pd.read_csv('./Data/20150701094451-Video_casts.csv')
# we don't care about these for now
casts = casts.drop('country', 1).drop('gender', 1)
behaviors = pd.read_csv('./Data/20150701094451-Behavior_training.csv',dtype={'user_id':pd.np.string_,'score':pd.np.string_})
behaviors = behaviors.drop('date_hour', 1).drop('mv_ratio', 1)

# combined all casts to join with videos
print "=> Pre-processing Casts and Behaviors into Videos"
print datetime.datetime.now()
casts = casts.groupby('container_id',as_index=False).agg(lambda x: ' '.join(x.person_id))
videos = pd.merge(videos, casts, on=['container_id'], how='left', suffixes=['_left', '_right'])

# combined ppl who watched this video (and their "scores")
behaviors = behaviors.groupby('video_id',as_index=False).agg(lambda x: ' '.join(x.user_id + '_' + x.score)).drop('score', 1)
videos = pd.merge(videos, behaviors, on=['video_id'], how='left', suffixes=['_left', '_right'])

## ==================== Features similarity
print "=> Features similarity"
print datetime.datetime.now()
videos['dummy'] = 1
videos_matrix = pd.merge(videos, videos, on=['dummy'], suffixes=['_left', '_right'])
videos_matrix = videos_matrix.drop('dummy', 1)

print "===> Country"
print datetime.datetime.now()
# country similarity: 0/1
def sim_country(row):
    return (1 if row['origin_country_left'] == row['origin_country_right'] else 0)

videos_matrix['sim_country'] = videos_matrix.apply(sim_country, axis=1)

print "===> Language"
print datetime.datetime.now()
# language similarity: 0/1
def sim_language(row):
    return (1 if row['origin_language_left'] == row['origin_language_right'] else 0)

videos_matrix['sim_language'] = videos_matrix.apply(sim_language, axis=1)

print "===> Adult"
print datetime.datetime.now()
# adult similarity: 0/1
def sim_adult(row):
    return (1 if row['adult_left'] == row['adult_right'] else 0)

videos_matrix['sim_adult'] = videos_matrix.apply(sim_adult, axis=1)

print "===> Content Owner"
print datetime.datetime.now()
# content_owner_id similarity: 0/1
def sim_content_owner_id(row):
    return (1 if row['content_owner_id_left'] == row['content_owner_id_right'] else 0)

videos_matrix['sim_content_owner_id'] = videos_matrix.apply(sim_content_owner_id, axis=1)

print "===> Broascast Time"
print datetime.datetime.now()
# broadcast_from, broadcast_to # date similarity
def sim_broadcast(row):
    try:
        bfl_date = datetime.datetime.strptime(row['broadcast_from_left'], "%Y-%m").date()
        bfr_date = datetime.datetime.strptime(row['broadcast_from_right'], "%Y-%m").date()
        btl_date = datetime.datetime.strptime(row['broadcast_to_left'], "%Y-%m").date()
        btr_date = datetime.datetime.strptime(row['broadcast_to_right'], "%Y-%m").date()
        return 1 / (abs((bfl_date-bfr_date).days) + abs((btl_date-btr_date).days))
    except:
        return 0

videos_matrix['sim_broadcast'] = videos_matrix.apply(sim_broadcast, axis=1)

print "===> Seasons"
print datetime.datetime.now()
# season_number
def sim_season(row):
    left = int(row['season_number_left'] if row['season_number_left'].isdigit() else 0)
    right = int(row['season_number_right'] if row['season_number_right'].isdigit() else 0)
    return abs(left - right)

videos_matrix['sim_season'] = videos_matrix.apply(sim_season, axis=1)

print "===> Episodes"
print datetime.datetime.now()
# episode_count
def sim_episode_count(row):
    return abs(row['episode_count_left'] - row['episode_count_right'])

videos_matrix['sim_episode_count'] = videos_matrix.apply(sim_episode_count, axis=1)

print "===> Genres"
print datetime.datetime.now()
# genres
def sim_genres(row):
    try:
        left = set(re.findall("\(*.g\)", row['genres_left']))
        right = set(re.findall("\(*.g\)", row['genres_right']))
        return len(left&right) / len(left|right)
    except:
        return 0

videos_matrix['sim_genres'] = videos_matrix.apply(sim_genres, axis=1)

print "===> Casts"
print datetime.datetime.now()
# casts / person_id
def sim_cast(row):
    try:
        left = set(row['person_id_left'].split())
        right = set(row['person_id_right'].split())
        return (0 if len(left|right) == 0 else len(left&right) / len(left|right))
    except:
        return 0

videos_matrix['sim_cast'] = videos_matrix.apply(sim_cast, axis=1)

## ==================== CF similarity
print "=> Calculating Jaccard indexes"
print datetime.datetime.now()
def jaccard(row):
    try:
        left_1 = set([item for item in row['user_id_left'].split() if item.endswith('_1')])
        right_1 = set([item for item in row['user_id_right'].split() if item.endswith('_1')])
        jaccard_1 = len(left_1&right_1) / len(left_1|right_1)
    except:   
        jaccard_1 = 0
    try:
        left_2 = set([item for item in row['user_id_left'].split() if item.endswith('_2')])
        right_2 = set([item for item in row['user_id_right'].split() if item.endswith('_2')])
        jaccard_2 = len(left_2&right_2) / len(left_2|right_2)
    except:   
        jaccard_2 = 0
    try:
        left_3 = set([item for item in row['user_id_left'].split() if item.endswith('_3')])
        right_3 = set([item for item in row['user_id_right'].split() if item.endswith('_3')])
        jaccard_3 = len(left_3&right_3) / len(left_3|right_3)
    except:
        jaccard_3 = 0
    
    return (10*jaccard_3 + 5*jaccard_2 + jaccard_1)

# This might take ~2.5 hours or more to finish.
videos_matrix['jaccard'] = videos_matrix.apply(jaccard, axis=1)

## ===================== Combined
# Feature scaling:
print "=> Feature scaling"
print datetime.datetime.now()
sim_features = ['sim_country','sim_language', 'sim_adult', 'sim_content_owner_id', 'sim_broadcast', 'sim_episode_count', 'sim_genres', 'sim_cast', 'jaccard']
weight_features = [10,10,5,1,1,1,5,5,20]
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

## Output to CSV
print "=> Out to CSV"
print datetime.datetime.now()
videos_matrix = videos_matrix.drop('origin_country_left', 1)
videos_matrix = videos_matrix.drop('origin_language_left', 1)
videos_matrix = videos_matrix.drop('adult_left', 1)
videos_matrix = videos_matrix.drop('content_owner_id_left', 1)
videos_matrix = videos_matrix.drop('broadcast_from_left', 1)
videos_matrix = videos_matrix.drop('broadcast_to_left', 1)
videos_matrix = videos_matrix.drop('season_number_left', 1)
videos_matrix = videos_matrix.drop('episode_count_left', 1)
videos_matrix = videos_matrix.drop('genres_left', 1)
videos_matrix = videos_matrix.drop('person_id_left', 1)
videos_matrix = videos_matrix.drop('user_id_left', 1)
videos_matrix = videos_matrix.drop('origin_country_right', 1)
videos_matrix = videos_matrix.drop('origin_language_right', 1)
videos_matrix = videos_matrix.drop('adult_right', 1)
videos_matrix = videos_matrix.drop('content_owner_id_right', 1)
videos_matrix = videos_matrix.drop('broadcast_from_right', 1)
videos_matrix = videos_matrix.drop('broadcast_to_right', 1)
videos_matrix = videos_matrix.drop('season_number_right', 1)
videos_matrix = videos_matrix.drop('episode_count_right', 1)
videos_matrix = videos_matrix.drop('genres_right', 1)
videos_matrix = videos_matrix.drop('person_id_right', 1)
videos_matrix = videos_matrix.drop('user_id_right', 1)
videos_matrix = videos_matrix.drop('container_id_left', 1)
videos_matrix = videos_matrix.drop('container_id_right', 1)
videos_matrix.to_csv("./data/videos_similarity_matrix.csv", sep='\t', encoding='utf-8', index=False)
print datetime.datetime.now()
