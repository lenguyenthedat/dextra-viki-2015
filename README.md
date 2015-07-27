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

    $ python viki-users-recommender.py # imperative approach - one by one - should take weeks to finish
    $ python viki-users-recommender-functional.py # functional - batch approach - 30 mins

This is more practical since `#videos x #videos` matrix is much smaller.
Weights can be set manually:

    sim_features = ['sim_country','sim_language', 'sim_adult', 'sim_content_owner_id', 'sim_broadcast', 'sim_episode_count', 'sim_genres', 'sim_cast', 'jaccard']
    weight_features = [10,10,10,3,3,3,10,10,50]

Submission history:

- Standard scaler with weight_features 10-10-05-01-01-01-05-05-20 Result: 0.1459
- Standard scaler with weight_features 10-10-05-01-01-01-02-02-10 Result: 0.135167
- Standard scaler with weight_features 10-10-10-03-03-03-10-10-40 Result: 0.145651
- Standard scaler with weight_features 10-10-10-03-03-03-10-10-10 Result: 0.0877144 => Jaccard is pretty good!
- Min-Max  scaler with weight_features 10-10-10-03-03-03-10-10-50 Result: 0.099632
- Min-Max  scaler with weight_features 10-05-05-03-03-03-05-05-20 Result: 0.0897612 => MinMax scaler doesn't really work very well does it?
- Standard scaler with weight_features 05-05-05-01-01-01-05-05-50 Result: 0.158367