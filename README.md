viki-challenge
==============

http://www.dextra.sg/rakuten-viki-global-tv-recommender-challenge/
https://challenges.dextra.sg/challenge/43

# Requirements:
This solution is 100% Python, below are a few libraries needed:

- Pandas
- Scikit-learn, xgboost, scikit-neuralnetwork

# First version: Tradditional Regression model (NeuralNet / RandomForest / GradientBoosting / XGBoosting)

    $ python viki-challenge-pre-process.py
    $ python viki-challenge-ml.py

Doesn't work very well, running on real test set (`#users x #videos` matrix) took forever.
Trained model can achieve RMSE ~ 48 (for `mv_ratio` as goal).

# Second version: Collaborative Filtering (Jaccard Index) plus feature similarity

    $ python viki-videos-similarity.py # Pre-procesing `#videos x #videos` matrix, roughly 2.5 hours

Then run recommender with either one of these

    $ python viki-users-recommender.py # batch process - 30 mins

This is more practical since `#videos x #videos` matrix is much smaller.
Weights can be set manually:

    sim_features    = ['sim_gender', 'sim_country', 'sim_language',
                       'sim_adult', 'sim_content_owner_id', 'sim_broadcast',
                       'sim_episode_count', 'sim_genres', 'sim_cast',
                       'jaccard', jaccard_1', 'jaccard_2', 'jaccard_3']
    weight_features = [5,5,10,
                       10,1,3,
                       3,5,10,
                       10,10,25,100]

Submission history:

- Standard scaler with weights: 10-10-05-01-01-01-05-05-20           Result: 0.1459
- Standard scaler with weights: 10-10-05-01-01-01-02-02-10           Result: 0.135167
- Standard scaler with weights: 10-10-10-03-03-03-10-10-40           Result: 0.145651
- Standard scaler with weights: 10-10-10-03-03-03-10-10-10           Result: 0.0877144 => Jaccard is pretty good!
- Min-Max  scaler with weights: 10-10-10-03-03-03-10-10-50           Result: 0.099632
- Min-Max  scaler with weights: 10-05-05-03-03-03-05-05-20           Result: 0.0897612 => MinMax scaler doesn't really work very well does it?
- Standard scaler with weights: 05-05-05-01-01-01-05-05-50           Result: 0.158367
- Standard scaler with weights: 05-05-05-01-01-01-05-05-50           Result: 0.15737 (Tried hot_videos_m and hot_videos_f)
- Standard scaler with weights: 05-05-05-01-01-01-05-10-80           Result: 0.160277
- Standard scaler with weights: 00-00-00-00-00-00-00-00-01           Result: 0.1611 (jaccard alone)
- Standard scaler with weights: 00-00-00-00-00-00-00-00-00-01-01-01  Result: 0.140996 - Added sim_gender & separated 3 jaccard scores.
- Min-Max  scaler with weights: 00-00-00-00-00-00-00-00-00-01-05-25  Result: 0.15947
- Standard scaler with weights: 05-05-10-10-01-03-03-05-10-10-25-100 Result: 0.16113 - Better scaling and feature similarity calculation
- Standard scaler with weights: 05-05-10-10-01-03-03-05-10-10-30-90  Result: 0.155238 - Try (x,1), (x,2) and (x,3) score grouping instead of (x,x)
- Standard scaler with weights: 05-05-10-10-01-03-03-05-10-15-30-45  Result: 0.150099
- Standard scaler with weights: 05-05-10-10-01-03-03-05-10-0-100-200 Result: 0.152626
- Standard scaler with weights: 05-05-10-10-01-03-03-05-10-10-20-40  Result: 0.151006
- Standard scaler with weights: 00-00-00-00-00-00-00-00-00-10-50-100 Result: 0.15191 - Again, just Jaccard
- Standard scaler with weights: 05-05-10-10-01-03-03-05-10-10-25-100 Result: 0.159887 - Do not punish low similarity (only count when score is > 0)
- Standard scaler with weights: 05-05-10-10-01-03-03-05-10-25-10-50-100 Result: 0.161194 - additional original jaccard + old 3 jaccard
TODO:
- Figuring out country restriction problem
- Better weights for movies watched score instead of 1,2,3?
