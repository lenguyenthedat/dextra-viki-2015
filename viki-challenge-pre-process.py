import pandas as pd
from sklearn.preprocessing import LabelEncoder

pd.options.mode.chained_assignment = None

features_non_numeric = ['country','gender','container_id', 'origin_country',
                        'origin_language','adult','broadcast_from','broadcast_to',
                        'season_number','content_owner_id','genres']

## Data preparation
videos = pd.read_csv('./data/20150701094451-Video_attributes.csv')
users = pd.read_csv('./data/20150701094451-User_attributes.csv')
behaviors = pd.read_csv('./data/20150701094451-Behavior_training.csv')
test = pd.read_csv('./data/20150701094451-Sample_submission.csv')
test_users = pd.DataFrame({'user_id' : test['user_id'][test['user_id'] > 0].unique()})

# Pre-processing non-number values
le = LabelEncoder()
for col in features_non_numeric:
    if col in videos.columns: le.fit(list(videos[col]))
    if col in videos.columns: videos[col] = le.transform(videos[col])

le = LabelEncoder()
for col in features_non_numeric:
    if col in users.columns: le.fit(list(users[col]))
    if col in users.columns: users[col]  = le.transform(users[col])

le = LabelEncoder()
for col in features_non_numeric:
    if col in behaviors.columns: le.fit(list(behaviors[col]))
    if col in behaviors.columns: behaviors[col] = le.transform(behaviors[col])

# Training data
master = pd.merge(behaviors, users, on='user_id', suffixes=['_left', '_right'])
master = pd.merge(master, videos, on='video_id', suffixes=['_left', '_right'])
master = master.drop('date_hour', 1)
master.to_csv("./data/train.csv", encoding='utf-8', index=False)

# Test data
# Added a 'dummy' column to mass join users & hot_videos into a matrix
videos['dummy'] = 1
users['dummy'] = 1
# only consider top 100 - performacne purpose
hot_videos = pd.DataFrame({'video_id':behaviors.groupby('video_id').agg(['count']).sort([('date_hour', 'count')], ascending=False).head(100).index.tolist()})
hot_videos = pd.merge(hot_videos, videos, on=['video_id'], suffixes=['_left', '_right'])

test_master = pd.merge(users, test_users, on='user_id', suffixes=['_left', '_right'])
test_master = pd.merge(test_master, hot_videos, on='dummy', suffixes=['_left', '_right'])
test_master.to_csv("./data/test.csv", encoding='utf-8', index=False)
