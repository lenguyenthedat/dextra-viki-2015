viki-challenge
==============
![img](http://i.imgur.com/LWnqxzM.png)

http://www.dextra.sg/rakuten-viki-global-tv-recommender-challenge/

https://challenges.dextra.sg/challenge/43

# Some preliminary analysis:
https://public.tableau.com/profile/le.nguyen.the.dat#!/vizhome/Rakuten-VikiDataScienceChallenge2015/Rakuten-VikiDataScienceChallenge2015

# Presentation deck:
https://speakerdeck.com/lenguyenthedat/rakuten-viki-data-challenge-solution

# Requirements:
This solution is 100% Python, below are a few libraries needed:

- Pandas
- Scikit-learn

# Collaborative Filtering (Jaccard Index) plus feature similarity

    $ python viki-videos-similarity.py # Pre-procesing `#videos x #videos` matrix
    $ python viki-users-recommender.py # batch process

This is more practical since `#videos x #videos` matrix is much smaller.
Weights can be set manually:

    top_videos_limit = 50
    sim_features = ['sim_country', 'sim_language', 'sim_adult',
                    'sim_content_owner_id', 'sim_broadcast', 'sim_episode_count',
                    'sim_genres', 'sim_cast',
                    'jaccard_1_3', 'jaccard_2_3', 'jaccard_3_3',
                    'jaccard_high', 'sim_cosine_mv_ratio']
    weight_features = [3,3,5,
                       0,0,0,
                       5,5,
                       0,0,15,
                       15,45]
    weight_scores = [1,3,15]

Notes:
------
The reason why HOT VIDEOS dominated CF is because viki's homepage currently dominated by:
- Top banner
- Popular show
- Top Drama
- # Gender filter for male / female

TODO:
-----
- utilize ratio instead of score
- see if someone is into hot / fresh video or not
- KNN: cosine similarity for user => top 10 similar user => recommend top videos user havent watched

Tried implementing with cosinesimilarity - Killed 9

Tried Sklearn KNN - took 5h ++

Panns https://github.com/ryanrhymes/panns 2h++

Trying Spotify's Annoy https://github.com/spotify/annoy : 20mins with 10 trees, 12mins with 100 trees.

-> however it's taking too long to find k-NN for each users (more than 10s each to get a good enough result)

Submission history:
-------------------
(Only those that worth documented or created when I am not too lazy):

https://github.com/lenguyenthedat/dextra-viki-2015/blob/master/submission_history.txt
