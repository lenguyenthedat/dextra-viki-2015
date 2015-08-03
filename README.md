viki-challenge
==============

http://www.dextra.sg/rakuten-viki-global-tv-recommender-challenge/
https://challenges.dextra.sg/challenge/43

# Some preliminary analysis:
https://public.tableau.com/profile/le.nguyen.the.dat#!/vizhome/Rakuten-VikiDataScienceChallenge2015/Rakuten-VikiDataScienceChallenge2015

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

1- Standard scaler with weights: 10-10-05-01-01-01-05-05-20           Result: 0.1459
2- Standard scaler with weights: 10-10-05-01-01-01-02-02-10           Result: 0.135167
3- Standard scaler with weights: 10-10-10-03-03-03-10-10-40           Result: 0.145651
4- Standard scaler with weights: 10-10-10-03-03-03-10-10-10           Result: 0.0877144 => Jaccard is pretty good!
5- Min-Max  scaler with weights: 10-10-10-03-03-03-10-10-50           Result: 0.099632
6- Min-Max  scaler with weights: 10-05-05-03-03-03-05-05-20           Result: 0.0897612 => MinMax scaler doesn't really work very well does it?
7- Standard scaler with weights: 05-05-05-01-01-01-05-05-50           Result: 0.158367
8- Standard scaler with weights: 05-05-05-01-01-01-05-05-50           Result: 0.15737 (Tried hot_videos_m and hot_videos_f)
9- Standard scaler with weights: 05-05-05-01-01-01-05-10-80           Result: 0.160277
10- Standard scaler with weights: 00-00-00-00-00-00-00-00-01           Result: 0.1611 (jaccard alone)
11- Standard scaler with weights: 00-00-00-00-00-00-00-00-00-01-01-01  Result: 0.140996 - Added sim_gender & separated 3 jaccard scores.
12- Min-Max  scaler with weights: 00-00-00-00-00-00-00-00-00-01-05-25  Result: 0.15947
13- Standard scaler with weights: 05-05-10-10-01-03-03-05-10-10-25-100 Result: 0.16113 - Better scaling and feature similarity calculation
14- Standard scaler with weights: 05-05-10-10-01-03-03-05-10-10-30-90  Result: 0.155238 - Try (x,1), (x,2) and (x,3) score grouping instead of (x,x)
15- Standard scaler with weights: 05-05-10-10-01-03-03-05-10-15-30-45  Result: 0.150099
16- Standard scaler with weights: 05-05-10-10-01-03-03-05-10-0-100-200 Result: 0.152626
17- Standard scaler with weights: 05-05-10-10-01-03-03-05-10-10-20-40  Result: 0.151006
18- Standard scaler with weights: 00-00-00-00-00-00-00-00-00-10-50-100 Result: 0.15191 - Again, just Jaccard
19- Standard scaler with weights: 05-05-10-10-01-03-03-05-10-10-25-100 Result: 0.159887 - Do not punish low similarity (only count when score is > 0)
20- Standard scaler with weights: 05-05-10-10-01-03-03-05-10-25-10-50-100 Result: 0.161194 - additional original jaccard + old 3 jaccard
21- Standard scaler with weights: 05-05-10-10-01-03-03-05-10-25-10-75-195 Result: 0.160826
22- Standard scaler with weights: 05-05-10-10-01-03-03-05-10-10-10-50-100 Result: 0.156655
23- Standard scaler with weights: 03-10-10-20-10-03-03-10-20-10-10-50-100 Result: 0.162476 - better weights
24- Standard scaler with weights: 03-10-10-20-10-03-03-10-20-10-10-50-100-01-05-10 Result: 0.161676 - 1-2-3 seems better...
25- Standard scaler with weights: 03-10-10-20-10-03-03-10-20-10-10-50-100-01-01-01 Result: 0.159341 - 1-2-3 seems best...
26- Standard scaler with weights: 03-10-10-20-10-03-03-10-20-10-10-50-100-01-02-03 Result: 0.162476
27- Standard scaler with weights: 03-10-10-20-10-03-03-10-20-10-10-50-100-01-01-01 Result: 0.161225 - 1-1-1 but use mv_ratio too
28- Standard scaler with weights: 03-95-95-20-10-03-03-10-20-10-10-50-100-01-02-03 Result: 0.161059
29- Standard scaler with weights: 03-10-10-20-10-03-03-10-20-50-50-01-02-03        Result: 0.135354 - cosine similarity instead of jaccard
30- Standard scaler with weights: 03-10-10-20-10-03-03-10-20-100-00-01-02-03       Result: 0.148153 - cosine similarity (mv ratio) instead of jaccard
31- Standard scaler with weights: 03-10-10-20-10-03-03-10-20-00-100-01-02-03       Result: 0.125421 - cosine similarity (score) instead of jaccard
32- Standard scaler with weights: 00-00-00-00-00-00-00-00-00-100-00-01-02-03       Result: 0.148942 - cosine similarity (mv ratio) alone
33- Standard scaler with weights: 00-00-00-00-00-00-00-00-00-00-00-00-50-50-50-01-02-03 Result: 0.168689 - New Jaccard and only itself
34- Standard scaler with weights: 03-10-10-20-10-03-03-10-20-00-00-00-50-50-50-01-02-03 Result: 0.165453 - New Jaccard with other weights
35- Standard scaler with weights: 00-00-00-00-00-00-00-00-00-00-00-00-50-50-50-25-25-25-01-02-03  Result: 0.152911 - New Jaccard and only itself (wrong weightages)
36- Standard scaler with weights: 00-00-00-00-00-00-00-00-00-00-00-00-15-15-15-50-50-50-01-02-03  Result: 0.161728 - New Jaccard and only itself
37- Standard scaler with weights: 00-00-00-00-00-00-00-00-00-50-50-50-01-02-03 Result: 0.170228 - new hot video formula (#user / time^2)
38- Standard scaler with weights: 00-00-00-00-00-00-00-00-00-50-50-50-01-02-03 Result: 0.172902 - new hot video formula (#user / time)
39- Standard scaler with weights: 00-00-00-00-00-00-00-00-15-50-50-50-01-02-03 Result: 0.188361 - Weight hot videos higher.
40- Standard scaler with weights: 00-00-00-00-00-00-00-00-05-50-50-50-01-02-03 Result: 0.175524 - Well, hot videos are really important.
41- Only hot videos                                                            Result: 0.197173 - Damn
42- Only hot videos (filter by score==3)                                       Result: 0.187228
43- 00-00-00-00-00-00-00-00-50-50-50-50-01-02-03 Result: 0.194124 - Refactoring
44- 50-00-00-00-00-00-00-00-50-10-25-50-01-02-03 Result: 0.191452
45- 00-00-00-00-00-00-00-00-50-10-25-50-01-02-03 Result: 0.194124
46- 01-00-00-00-00-00-00-00-01-00-00-01-01-02-03 Result: 0.191452
47- 00-00-00-00-00-00-00-00-00-05-15-35-01-02-03 Result: 0.181584 - limit video-matrix to include top 20 videos only (bugged)
48- 00-00-00-00-00-00-00-00-00-00-00-01-01-02-03 Result: 0.181584 - bugged
49- 00-00-00-00-00-00-00-00-00-00-00-01-01-02-03 Result: 0.176865 - filtering bug
50- 00-00-00-00-00-00-00-00-01-01-03-05-01-02-03 Result: 0.181028 - filtering bug
51- 00-00-00-00-00-00-00-00-00-01-03-05-01-02-03 Result: 0.176821 - jaccard on only top 20 (unless user already viewed all of them) bug fixed
52- 00-00-00-00-00-00-00-00-00-00-00-01-01-02-03 Result: 0.179890 - jaccard_3 on only top 10 (unless user already viewed all of them) bug fixed

# TODO
- Figuring out country restriction problem -> might not be feasible for this challenge, since we do not know if a user is premium or not (premium users doesn't get regional blocked)
- Time decay function -> might not be feasible for this challenge, since training data is all in a span of 3-4 months.
