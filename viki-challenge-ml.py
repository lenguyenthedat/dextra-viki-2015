import pandas as pd
import time
import csv
import numpy as np
import os

from sklearn.metrics import mean_squared_error, make_scorer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sknn.mlp import Regressor, Layer
from sklearn.grid_search import GridSearchCV
from xgboost import XGBRegressor

pd.options.mode.chained_assignment = None

sample = False
gridsearch = False

features = ['user_id', 'video_id', 'country', 'gender', 'container_id',
            'origin_country', 'origin_language', 'adult', 'broadcast_from', 'broadcast_to',
            'season_number', 'content_owner_id', 'genres', 'episode_count']
features_non_numeric = ['country','gender','container_id', 'origin_country',
                        'origin_language','adult','broadcast_from','broadcast_to',
                        'season_number','content_owner_id','genres']

goal = 'score' #1-3

# Load data
if sample: # To train with 75% data
    df = pd.read_csv('./data/train.csv')
    df['is_train'] = np.random.uniform(0, 1, len(df)) <= .75
    train, test = df[df['is_train']==True], df[df['is_train']==False]
else:
    # To run with real data
    train = pd.read_csv('./data/train.csv')
    test = pd.read_csv('./data/test.csv')

train['video_id'] = train['video_id'].map(lambda x: x.lstrip('TV'))
test['video_id'] = test['video_id'].map(lambda x: x.lstrip('TV'))

# Define regressors
if sample:
    regressors = [
        # Regressor(layers=[
        #             Layer("Sigmoid", units=100),
        #             Layer("Sigmoid", units=100),
        #             Layer("Linear")],
        #           learning_rate=0.01,learning_rule='adadelta',learning_momentum=0.9,
        #           batch_size=100,valid_size=0.01,
        #           n_stable=50,n_iter=200,verbose=True),
        GradientBoostingRegressor(n_estimators=10, learning_rate=1.0,max_depth=5, random_state=0),
        RandomForestRegressor(max_depth=8,n_estimators=128),
        XGBRegressor(max_depth=2,n_estimators=512)
    ]
else:
    regressors = [# Other methods are underperformed yet take very long training time for this data set
        GradientBoostingRegressor(n_estimators=10, learning_rate=1.0,max_depth=5, random_state=0)
        # RandomForestRegressor(max_depth=8,n_estimators=128),
        # XGBRegressor(max_depth=2,n_estimators=512)
    ]

# Train
for regressor in regressors:
    print regressor.__class__.__name__
    start = time.time()
    if (gridsearch & sample): # only do gridsearch if we run with sampled data.
        try: # depth & estimator: usually fits for RF and XGB
            if (regressor.__class__.__name__ == "GradientBoostingRegressor"):
                print "Attempting GridSearchCV for GB model"
                gscv = GridSearchCV(regressor, {
                    'max_depth': [2, 8, 16],
                    'n_estimators': [32, 64, 128, 256, 512],
                    'learning_rate': [0.1, 0.05, 0.01],
                    'subsample': [0.6,0.8,1]},
                    verbose=1, n_jobs=2, scoring=gini_scorer)
            if (regressor.__class__.__name__ == "XGBRegressor"):
                print "Attempting GridSearchCV for XGB model"
                gscv = GridSearchCV(regressor, {
                    'max_depth': [2, 8, 16],
                    'n_estimators': [32, 64, 128, 256, 512],
                    'min_child_weight': [3,5],
                    'subsample': [0.6,0.8,1]},
                    verbose=1, n_jobs=2, scoring=gini_scorer)
            if (regressor.__class__.__name__ == "RandomForestRegressor"):
                print "Attempting GridSearchCV for RF model"
                gscv = GridSearchCV(regressor, {
                    'max_depth': [2, 8, 16],
                    'n_estimators': [32, 64, 128, 256, 512],
                    'bootstrap':[True,False],
                    'oob_score': [True,False]},
                    verbose=1, n_jobs=2, scoring=gini_scorer)
            if (regressor.__class__.__name__ == "Regressor"): # NN Regressor
                print "Attempting GridSearchCV for Neural Network model"
                gscv = GridSearchCV(regressor, {
                    'hidden0__units': [4, 16, 64, 128],
                    'hidden0__type': ["Rectifier", "Sigmoid", "Tanh"]},
                    verbose=1, n_jobs=1)
            regressor = gscv.fit(np.array(train[list(features)]), train[goal])
            print(regressor.best_score_)
            print(regressor.best_params_)
        except:
            regressor.fit(np.array(train[list(features)]), train[goal]) # just fit regular one
    else:
        regressor.fit(np.array(train[list(features)]), train[goal])
    print '  -> Training time:', time.time() - start

# Evaluation and export result
if sample:
    # Test results
    for regressor in regressors:
        print regressor.__class__.__name__
        try:
            print 'Root mean_squared_error:'
            print mean_squared_error(test[goal],regressor.predict(np.array(test[features])))**0.5
        except:
            pass

else: # Export result
    count = 0
    for regressor in regressors:
        count += 1
        if not os.path.exists('result/'):
            os.makedirs('result/')
        # TODO: fix this shit
        # test[myid] values will get converted to float since column_stack will result in array
        predictions = np.column_stack((test['user_id'],test['video_id'], regressor.predict(np.array(test[features])))).tolist()
        predictions = [[int(i[0])] + i[1:] for i in predictions]
        csvfile = 'result/' + regressor.__class__.__name__ + '-'+ str(count) + '-submit.csv'
        with open(csvfile, 'w') as output:
            writer = csv.writer(output, lineterminator='\n')
            writer.writerow(['user_id','video_id',goal])
            writer.writerows(predictions)
