from __future__ import division

import pandas as pd
import time
import csv
import numpy as np
import datetime
import re
import warnings
warnings.filterwarnings("ignore")

def read_data():
    """ Read and pre-process data
        >>> videos_matrix = read_data()
        >>> videos_matrix[:3]
          video_id_left container_id_left origin_country_left origin_language_left  \
        0         TV001      Container001                  us                   en
        1         TV001      Container001                  us                   en
        2         TV001      Container001                  us                   en

          adult_left broadcast_from_left broadcast_to_left season_number_left  \
        0      False                None              None               None
        1      False                None              None               None
        2      False                None              None               None

          content_owner_id_left genres_left  \
        0        ContentOwner01        None
        1        ContentOwner01        None
        2        ContentOwner01        None

                                 ...                          origin_language_right  \
        0                        ...                                             en
        1                        ...                                             en
        2                        ...                                             zt

          adult_right broadcast_from_right broadcast_to_right season_number_right  \
        0       False                 None               None                None
        1       False              2013-06            2013-08                   3
        2       False              2012-07            2012-11                None

          content_owner_id_right                                       genres_right  \
        0         ContentOwner01                                               None
        1         ContentOwner02                            Action & Adventure (1g)
        2         ContentOwner03  Comedy (6g), Drama (9g), Idol Drama (1038g), R...

          episode_count_right                                    person_id_right  \
        0                   5                                                NaN
        1                  10                                                NaN
        2                  77  Cast0898 Cast0483 Cast1344 Cast1688 Cast0503 C...

                                               user_id_right
        0  189500_2 328741_2 579541_2 153183_2 151295_3 3...
        1                                           353674_3
        2  759744_1 379687_3 160301_1 159490_1 151124_1 1...
    """
    videos = pd.read_csv('./data/20150701094451-Video_attributes.csv')
    casts = pd.read_csv('./data/20150701094451-Video_casts.csv')
    behaviors = pd.read_csv('./data/20150701094451-Behavior_training.csv',dtype={'user_id':pd.np.string_,'score':pd.np.string_})
    # we don't care about these for now
    behaviors = behaviors.drop(['date_hour','mv_ratio'], 1)
    # flattening casts
    casts = casts.drop(['country','gender'], 1).groupby('container_id',as_index=False).agg(lambda x: ' '.join(x.person_id))
    videos = pd.merge(videos, casts, on=['container_id'], how='left', suffixes=['_left', '_right'])
    # combined ppl who watched this video (and their "scores")
    behaviors = behaviors.groupby('video_id',as_index=False).agg(lambda x: ' '.join(x.user_id + '_' + x.score)).drop('score', 1)
    videos = pd.merge(videos, behaviors, on=['video_id'], how='left', suffixes=['_left', '_right'])
    # Constructing videos_matrix
    videos['dummy'] = 1
    videos_matrix = pd.merge(videos, videos, on=['dummy'], suffixes=['_left', '_right'])
    videos_matrix = videos_matrix.drop('dummy', 1)
    return videos_matrix

def feature_similarity(videos_matrix):
    """ Calculating feature similarity for each pair of movies.
        >>> videos_matrix = feature_similarity(videos_matrix)
        >>> videos_matrix[:3]
          video_id_left                                       user_id_left  \
        0         TV001  189500_2 328741_2 579541_2 153183_2 151295_3 3...
        1         TV001  189500_2 328741_2 579541_2 153183_2 151295_3 3...
        2         TV001  189500_2 328741_2 579541_2 153183_2 151295_3 3...

          video_id_right                                      user_id_right  \
        0          TV001  189500_2 328741_2 579541_2 153183_2 151295_3 3...
        1          TV002                                           353674_3
        2          TV003  759744_1 379687_3 160301_1 159490_1 151124_1 1...

           sim_country  sim_language  sim_adult  sim_content_owner_id  sim_broadcast  \
        0            1             1          1                     1              0
        1            1             1          1                     0              0
        2            0             0          1                     0              0

           sim_season  sim_episode_count  sim_genres  sim_cast
        0           0           1.000000           0         0
        1           0           0.500000           0         0
        2           0           0.064935           0         0
    """
    # Country Similarity
    def sim_country(row):
        return (1 if row['origin_country_left'] == row['origin_country_right'] else 0)
    videos_matrix['sim_country'] = videos_matrix.apply(sim_country, axis=1)
    # Language similarity:
    def sim_language(row):
        return (1 if row['origin_language_left'] == row['origin_language_right'] else 0)
    videos_matrix['sim_language'] = videos_matrix.apply(sim_language, axis=1)
    # Adult similarity:
    def sim_adult(row):
        return (1 if row['adult_left'] == row['adult_right'] else 0)
    videos_matrix['sim_adult'] = videos_matrix.apply(sim_adult, axis=1)
    # Content_owner_id similarity: 0/1
    def sim_content_owner_id(row):
        return (1 if row['content_owner_id_left'] == row['content_owner_id_right'] else 0)
    videos_matrix['sim_content_owner_id'] = videos_matrix.apply(sim_content_owner_id, axis=1)
    # Broadcast_from, broadcast_to # date similarity
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
    # Season_number
    def sim_season(row):
        try:
            left = int(row['season_number_left'] if row['season_number_left'].isdigit() else 0)
            right = int(row['season_number_right'] if row['season_number_right'].isdigit() else 0)
            return min(left,right)/max(left,right)
        except:
            return 0
    videos_matrix['sim_season'] = videos_matrix.apply(sim_season, axis=1)
    # Episode_count
    def sim_episode_count(row):
        try:
            return min(row['episode_count_left'],row['episode_count_right'])/max(row['episode_count_left'],row['episode_count_right'])
        except:
            return 0
    videos_matrix['sim_episode_count'] = videos_matrix.apply(sim_episode_count, axis=1)
    # Genres
    def sim_genres(row):
        try:
            left = set(re.findall("\(*.g\)", row['genres_left']))
            right = set(re.findall("\(*.g\)", row['genres_right']))
            return len(left&right) / len(left|right)
        except:
            return 0
    videos_matrix['sim_genres'] = videos_matrix.apply(sim_genres, axis=1)
    # Casts
    def sim_cast(row):
        try:
            left = set(row['person_id_left'].split())
            right = set(row['person_id_right'].split())
            return len(left&right) / len(left|right)
        except:
            return 0
    videos_matrix['sim_cast'] = videos_matrix.apply(sim_cast, axis=1)
    return videos_matrix.drop(['container_id_left', 'origin_country_left', 'origin_language_left', 'adult_left',
     'broadcast_from_left', 'broadcast_to_left', 'season_number_left', 'content_owner_id_left', 'genres_left',
     'episode_count_left', 'person_id_left', 'container_id_right',
     'origin_country_right', 'origin_language_right', 'adult_right', 'broadcast_from_right',
     'broadcast_to_right', 'season_number_right', 'content_owner_id_right', 'genres_right',
     'episode_count_right', 'person_id_right'],1)

def jaccard_similarity(videos_matrix):
    """ Calculating jaccard similarity for each pair of movies.
        >>> videos_matrix = jaccard_similarity(videos_matrix)
        >>> videos_matrix[:3]
          video_id_left                                       user_id_left  \
        0         TV001  189500_2 328741_2 579541_2 153183_2 151295_3 3...
        1         TV001  189500_2 328741_2 579541_2 153183_2 151295_3 3...
        2         TV001  189500_2 328741_2 579541_2 153183_2 151295_3 3...

          video_id_right                                      user_id_right  \
        0          TV001  189500_2 328741_2 579541_2 153183_2 151295_3 3...
        1          TV002                                           353674_3
        2          TV003  759744_1 379687_3 160301_1 159490_1 151124_1 1...

           sim_country  sim_language  sim_adult  sim_content_owner_id  sim_broadcast  \
        0            1             1          1                     1              0
        1            1             1          1                     0              0
        2            0             0          1                     0              0

           sim_season  sim_episode_count  sim_genres  sim_cast  jaccard_1_3  \
        0           0           1.000000           0         0            0
        1           0           0.500000           0         0            0
        2           0           0.064935           0         0            0

           jaccard_2_3  jaccard_3_3
        0            0      1.00000
        1            0      0.00000
        2            0      0.00159
    """
    print "=> Calculating Jaccard indexes #1-3"
    print datetime.datetime.now()
    def jaccard_1_3(row): # people who do not like LEFT but like RIGHT
        try:
            left_1 = set([item for item in row['user_id_left'].split() if item.endswith('_1')])
            right_3 = set([item for item in row['user_id_right'].split() if item.endswith('_3')])
            if len(left_1|right_3) < 1000:
                return 0
            else:
                return len(left_1&right_3) / len(left_1|right_3)
        except:
            return 0
    videos_matrix['jaccard_1_3'] = videos_matrix.apply(jaccard_1_3, axis=1)
    print "=> Calculating Jaccard indexes #2-3"
    print datetime.datetime.now()
    def jaccard_2_3(row): # people who kind of like LEFT and like RIGHT
        try:
            left_2 = set([item for item in row['user_id_left'].split() if item.endswith('_2')])
            right_3 = set([item for item in row['user_id_right'].split() if item.endswith('_3')])
            if len(left_2|right_3) < 1000:
                return 0
            else:
                return len(left_2&right_3) / len(left_2|right_3)
        except:
            return 0
    videos_matrix['jaccard_2_3'] = videos_matrix.apply(jaccard_2_3, axis=1)
    print "=> Calculating Jaccard indexes #3-3"
    print datetime.datetime.now()
    def jaccard_3_3(row): # people who like LEFT and like RIGHT
        try:
            left_3 = set([item for item in row['user_id_left'].split() if item.endswith('_3')])
            right_3 = set([item for item in row['user_id_right'].split() if item.endswith('_3')])
            if len(left_3|right_3) < 1000:
                return 0
            else:
                return len(left_3&right_3) / len(left_3|right_3)
        except:
            return 0
    videos_matrix['jaccard_3_3'] = videos_matrix.apply(jaccard_3_3, axis=1)
    return videos_matrix.drop(['user_id_left','user_id_right'],1)

def output_videos_matrix_to_csv(videos_matrix):
    videos_matrix.to_csv("./data/videos_similarity_matrix.csv", encoding='utf-8', index=False)

def main():
    print "=> Processing data - " + str(datetime.datetime.now())
    videos_matrix = read_data()
    print "=> Calculating feature similarities - " + str(datetime.datetime.now())
    videos_matrix = feature_similarity(videos_matrix)
    print "=> Calculating jaccard similarities - " + str(datetime.datetime.now())
    videos_matrix = jaccard_similarity(videos_matrix)
    print "=> Output to csv - " + str(datetime.datetime.now())
    output_videos_matrix_to_csv(videos_matrix)

if __name__ == "__main__":
    main()