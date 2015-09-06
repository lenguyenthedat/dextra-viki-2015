viki-challenge
==============
![img](http://i.imgur.com/LWnqxzM.png)

http://www.dextra.sg/rakuten-viki-global-tv-recommender-challenge/
https://challenges.dextra.sg/challenge/43

# Some preliminary analysis:
https://public.tableau.com/profile/le.nguyen.the.dat#!/vizhome/Rakuten-VikiDataScienceChallenge2015/Rakuten-VikiDataScienceChallenge2015

# Requirements:
This solution is 100% Python, below are a few libraries needed:

- Pandas
- Scikit-learn

# Collaborative Filtering (Jaccard Index) plus feature similarity

    $ python viki-videos-similarity.py # Pre-procesing `#videos x #videos` matrix, roughly 1.5 hours
    $ python viki-users-recommender.py # batch process - 20 mins

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

Submission history (Only those that worth documented or created when I am not too lazy):
----------------------------------------------------------------------------------------

1-  0.145900Standard scaler with weights: 10-10-05-01-01-01-05-05-20

2-  0.135167Standard scaler with weights: 10-10-05-01-01-01-02-02-10

3-  0.145651Standard scaler with weights: 10-10-10-03-03-03-10-10-40

4-  0.087714Standard scaler with weights: 10-10-10-03-03-03-10-10-10 => Jaccard is pretty good!

5-  0.099632Min-Max  scaler with weights: 10-10-10-03-03-03-10-10-50

6-  0.089761Min-Max  scaler with weights: 10-05-05-03-03-03-05-05-20 2 => MinMax scaler doesn't really work very well does it?

7-  0.158367Standard scaler with weights: 05-05-05-01-01-01-05-05-50

8-  0.157370Standard scaler with weights: 05-05-05-01-01-01-05-05-50 (Tried hot_videos_m and hot_videos_f)

9-  0.160277Standard scaler with weights: 05-05-05-01-01-01-05-10-80

10- 0.161100 Standard scaler with weights: 00-00-00-00-00-00-00-00-01 (jaccard alone)

11- 0.140996 Standard scaler with weights: 00-00-00-00-00-00-00-00-00-01-01-01 - Added sim_gender & separated 3 jaccard scores.

12- 0.159470 Min-Max  scaler with weights: 00-00-00-00-00-00-00-00-00-01-05-25

13- 0.161130 Standard scaler with weights: 05-05-10-10-01-03-03-05-10-10-25-100  - Better scaling and feature similarity calculation

14- 0.155238 Standard scaler with weights: 05-05-10-10-01-03-03-05-10-10-30-90   - Try (x,1), (x,2) and (x,3) score grouping instead of (x,x)

15- 0.150099 Standard scaler with weights: 05-05-10-10-01-03-03-05-10-15-30-45

16- 0.152626 Standard scaler with weights: 05-05-10-10-01-03-03-05-10-0-100-200

17- 0.151006 Standard scaler with weights: 05-05-10-10-01-03-03-05-10-10-20-40

18- 0.151910 Standard scaler with weights: 00-00-00-00-00-00-00-00-00-10-50-100  - Again, just Jaccard

19- 0.159887 Standard scaler with weights: 05-05-10-10-01-03-03-05-10-10-25-100  - Do not punish low similarity (only count when score is > 0)

20- 0.161194 Standard scaler with weights: 05-05-10-10-01-03-03-05-10-25-10-50-100  - additional original jaccard + old 3 jaccard

21- 0.160826 Standard scaler with weights: 05-05-10-10-01-03-03-05-10-25-10-75-195

22- 0.156655 Standard scaler with weights: 05-05-10-10-01-03-03-05-10-10-10-50-100

23- 0.162476 Standard scaler with weights: 03-10-10-20-10-03-03-10-20-10-10-50-100  - better weights

24- 0.161676 Standard scaler with weights: 03-10-10-20-10-03-03-10-20-10-10-50-100-01-05-10  - 1-2-3 seems better...

25- 0.159341 Standard scaler with weights: 03-10-10-20-10-03-03-10-20-10-10-50-100-01-01-01  - 1-2-3 seems best...

26- 0.162476 Standard scaler with weights: 03-10-10-20-10-03-03-10-20-10-10-50-100-01-02-03

27- 0.161225 Standard scaler with weights: 03-10-10-20-10-03-03-10-20-10-10-50-100-01-01-01  - 1-1-1 but use mv_ratio too

28- 0.161059 Standard scaler with weights: 03-95-95-20-10-03-03-10-20-10-10-50-100-01-02-03

29- 0.135354 Standard scaler with weights: 03-10-10-20-10-03-03-10-20-50-50-01-02-03 - cosine similarity instead of jaccard

30- 0.148153 Standard scaler with weights: 03-10-10-20-10-03-03-10-20-100-00-01-02-03 - cosine similarity (mv ratio) instead of jaccard

31- 0.125421 Standard scaler with weights: 03-10-10-20-10-03-03-10-20-00-100-01-02-03 - cosine similarity (score) instead of jaccard

32- 0.148942 Standard scaler with weights: 00-00-00-00-00-00-00-00-00-100-00-01-02-03 - cosine similarity (mv ratio) alone

33- 0.168689 Standard scaler with weights: 00-00-00-00-00-00-00-00-00-00-00-00-50-50-50-01-02-03 - New Jaccard and only itself

34- 0.165453 Standard scaler with weights: 03-10-10-20-10-03-03-10-20-00-00-00-50-50-50-01-02-03 - New Jaccard with other weights

35- 0.152911 Standard scaler with weights: 00-00-00-00-00-00-00-00-00-00-00-00-50-50-50-25-25-25-01-02-03 - New Jaccard and only itself (wrong weightages)

36- 0.161728 Standard scaler with weights: 00-00-00-00-00-00-00-00-00-00-00-00-15-15-15-50-50-50-01-02-03 - New Jaccard and only itself

37- 0.170228 Standard scaler with weights: 00-00-00-00-00-00-00-00-00-50-50-50-01-02-03  - new hot video formula (#user / time^2)

38- 0.172902 Standard scaler with weights: 00-00-00-00-00-00-00-00-00-50-50-50-01-02-03  - new hot video formula (#user / time)

39- 0.188361 Standard scaler with weights: 00-00-00-00-00-00-00-00-15-50-50-50-01-02-03  - Weight hot videos higher.

40- 0.175524 Standard scaler with weights: 00-00-00-00-00-00-00-00-05-50-50-50-01-02-03  - Well, hot videos are really important.

41- 0.197173 Only hot videos - Damn

42- 0.187228 Only hot videos (filter by score==3)

43- 0.194124 00-00-00-00-00-00-00-00-50-50-50-50-01-02-03  - Refactoring

44- 0.191452 50-00-00-00-00-00-00-00-50-10-25-50-01-02-03

45- 0.194124 00-00-00-00-00-00-00-00-50-10-25-50-01-02-03

46- 0.191452 01-00-00-00-00-00-00-00-01-00-00-01-01-02-03

47- 0.181584 00-00-00-00-00-00-00-00-00-05-15-35-01-02-03  - limit video-matrix to include top 20 videos only (bugged)

48- 0.181584 00-00-00-00-00-00-00-00-00-00-00-01-01-02-03  - bugged

49- 0.176865 00-00-00-00-00-00-00-00-00-00-00-01-01-02-03  - filtering bug

50- 0.181028 00-00-00-00-00-00-00-00-01-01-03-05-01-02-03  - filtering bug

51- 0.176821 00-00-00-00-00-00-00-00-00-01-03-05-01-02-03  - jaccard on only top 20 (unless user already viewed all of them) bug fixed

52- 0.179890 00-00-00-00-00-00-00-00-00-00-00-01-01-02-03  - jaccard_3 on only top 10 (unless user already viewed all of them) bug fixed

53- 0.189109 00-00-00-00-00-00-00-00-01-03-05-01-02-03  (top 100 only) - factor in hotness properly

54- 0.190801 00-00-00-00-00-00-00-00-01-03-05-01-02-03  (top 20 only)

55- 0.192084 00-00-00-00-00-00-00-00-01-03-05-01-02-03  (top 10 only)

56- 0.191566 01-01-01-00-00-00-01-01-00-01-03-00-01-03

57- 0.201594 new hot videos

58- 0.202451 new hot videos (only count score= 2 or 3)

59- 0.181191 new hot videos (score= 2 or 3, sqrt)

60- 0.189585 new hot videos (score= 2 or 3, pow 2)

61- 0.199957 01-01-01-00-00-00-01-01-00-04-07-00-01-02-top-20

62- 0.199543 03-03-05-00-00-00-05-05-00-02-10-01-02-03-top-20

63- 0.202127 03-03-05-00-00-00-05-05-00-02-10-01-02-03-top-20   - freshness

64- 0.204343 03-03-05-00-00-00-05-05-00-02-10-01-02-03-top-10

65- 0.202351 03-03-05-00-00-00-05-05-00-02-10-01-02-03-top-100

66- 0.206155 03-30-05-00-00-00-05-05-01-05-25-01-05-25-top-10

67- 0.204596 03-03-05-00-00-00-05-15-01-15-45-01-05-45-top-10

68- 0.206163 03-03-05-00-00-00-05-05-01-05-45-01-05-15-top-10

69- 0.201184 03-03-05-00-00-00-05-05-00-05-45-00-05-15-top-100  - hotness^2

70- 0.195498 03-03-05-00-00-00-05-05-00-05-05-00-05-15-top-50

71- 0.201141 00-00-00-00-00-00-00-00-00-15-45-00-05-15-top-50

72- 0.204553 00-00-00-00-00-00-00-00-00-15-45-00-05-15-top-10   Minmax

73- 0.201493 03-03-05-00-00-00-05-05-00-05-45-00-05-15-top-100  - minmax freshness 1-2

74- 0.203345 03-03-05-00-00-00-05-05-01-05-45-01-05-15-top-10   No freshness

75- 0.205918 03-03-05-00-00-00-05-05-01-05-45-01-05-15-top-10   freshness^2

76- 0.205935 03-03-05-00-00-00-05-05-01-05-45-01-05-15-top-10   freshness^2 minmax 1-2

77- 0.168555 03-03-05-00-00-00-05-05-01-05-45-01-05-15-top-10   amazon-limit-by-top-perf

78- 0.184913 03-03-05-00-00-00-05-05-01-05-45-01-05-15-top-10   limit -by-hotness-combine-by-aws-perf

79- 0.202482 03-03-05-00-00-00-05-05-01-05-45-01-05-15-top-10   hotness only when score = 3

80- 0.203116 3-3-0-0-0-0-5-25-1-15-45-1-15-45-top-10

81- 0.204379 3-3-25-0-0-0-5-1-1-15-45-1-15-45-top-5

82- 0.199439 3-3-25-0-0-0-5-25-1-15-45-1-15-45-top-5-nofresh-nohotness

83- 0.184235 3-3-25-0-0-0-5-25-1-15-45-1-15-45-top-25-nohotness

84- 0.202711 90-5-5-0-0-0-5-15-1-10-45-1-5-7-top-100 -hot^2

85- 0.202381 99-5-5-0-0-0-5-15-1-45-45-1-5-5-top-25

86- 0.168672 99-5-5-0-0-0-5-15-1-45-45-1-5-5-top-100 -hotnessminmax1-2-newjaccard

87- 0.180541 99-5-5-0-0-0-5-15-1-45-45-1-5-5-top-100 -hotnessminmax0-1-newjaccard

88- 0.188778 99-5-5-0-0-0-5-15-1-45-45-1-5-5-top-100 -noscale-newjaccard

89- 0.194169 3-3-5-0-0-0-5-5-1-5-45-1-5-15-top-10 -newjaccard is bad

90- 0.206163 3-3-5-0-0-0-5-5-1-5-45-1-5-15-top-10

91- 0.205861 3-3-25-0-0-0-5-0-1-25-45-1-5-5-top-10

92- 0.199735 25-5-25-0-0-0-15-15-0-0-45-0-0-5-top-10

93- 0.204245 25-5-25-0-0-0-0-0-0-45-45-0-5-5-top-10

94- 0.211681 03-03-05-00-00-00-05-05-01-05-45-01-05-15-top-10   mimic #68 with a split for Hotness (male / female / overall)

95- 0.201120 00-00-00-00-00-00-00-00-00-00-00-01-05-15-top-10   hotness (split) only

96- 0.210312 03-03-05-00-00-00-05-05-01-05-45-01-05-15-top-10   mimic #94 with hotness^2

97- 0.211086 03-03-05-00-00-00-05-05-01-05-45-01-05-15-top-25   mimic #94 with top25

98- 0.212555 03-03-05-00-00-00-05-05-01-05-45-01-05-15-top-10   best_videos to be divided (f,m,o)

99- 0.212485 03-03-05-00-00-00-05-05-01-25-45-01-02-03-top-10

100 0.212396 03-03-05-00-00-00-05-05-01-05-45-01-05-15-top-10   #mimic 98 but treat hot_overall as female

101 0.214330 03-03-05-00-00-00-05-05-01-05-45-01-05-15-top-10   #fresh^2

102 0.213760 03-03-05-00-00-00-05-05-01-05-45-01-05-15-top-10   #fresh^4

103 0.212328 15-03-05-00-00-00-15-15-01-05-45-01-05-15-top-15

104 0.214434 03-03-05-00-00-00-05-05-01-05-45-01-05-15-top-10   #fresh^2, updated hot_o

105 0.213791 03-03-05-00-00-00-05-05-01-05-45-01-05-15-top-10   bestness

106 0.189606 03-03-05-00-00-00-05-05-01-05-45-01-05-15-top-10   bestness no hotness

108 0.214066 03-03-05-00-00-00-05-05-15-15-45-01-02-03-top-10

109 0.214225 03-03-05-00-00-00-05-05-25-25-25-01-05-15-top-10   fresh^3

110 0.214307 03-03-05-00-00-00-05-05-01-05-45-01-05-15-top-10   fresh^3

111 0.194942 svd-v1

112 0.214213 submit-3-3-5-0-0-0-5-5-1-5-45-95-1-5-15-top-10

113 0.213590 submit-3-3-5-0-0-0-5-5-1-5-25-25-1-2-3-top-10

114 0.212152 submit-0-0-0-0-0-0-0-0-0-1-1-1-0-1-1-top-10

115 0.205756 submit-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-top-10

116 0.214295 submit-3-3-5-0-0-0-5-5-1-5-45-25-1-5-15-top-10

117 0.208086 submit-5-3-0-0-0-0-5-15-1-10-25-25-0-1-5-top-15  no punishment if score less than 0, fixed jaccard to exclude not-hot movies

118 0.214475 submit-3-3-5-0-0-0-5-5-1-5-45-95-1-5-15-top-10  # old params

119 0.214270 submit-3-3-5-0-0-0-5-5-1-5-45-0-1-5-15-top-10

120 0.214474 submit-3-3-5-0-0-0-5-5-1-5-95-145-1-5-15-top-10

121 0.212959 submit-3-3-5-0-0-0-5-5-1-5-45-95-1-5-15-top-10

122 0.214205 submit-3-3-5-0-0-0-5-5-1-5-25-25-1-5-25-top-10

123 0.209646 submit-3-3-5-0-0-0-5-5-1-5-45-95-1-5-15-top-10-minmax

124 0.212232 submit-3-3-5-0-0-0-5-5-1-5-15-15-1-5-15-top-10-minmax

125 0.209993 submit-3-3-5-0-0-0-5-5-1-5-10-10-1-5-15-top-10-minmax

126 0.213060 submit-3-3-5-0-0-0-5-5-1-5-25-25-1-5-15-top-10-minmax

127 0.214074 submit-3-3-5-0-0-0-5-5-1-5-45-1-5-15-top-10-no-svd

128 0.212358 submit-3-3-5-0-0-0-5-5-1-5-15-1-5-15-top-100  - filter before scaling.

129 0.204905 submit-3-3-5-0-0-0-5-5-1-5-15-1-5-15-top-10

130 0.212648 submit-3-3-5-0-0-0-5-5-1-5-25-1-5-15-top-50

131 0.204847 submit-3-3-5-0-0-0-5-5-1-5-10-1-3-5-top-25

132 0.212818 submit-3-3-5-0-0-0-5-5-1-5-45-1-3-15-top-25

133 0.213767 submit-3-3-5-0-0-0-5-5-1-5-45-1-3-15-top-50

134 0.213771 submit-3-3-5-0-0-0-5-5-1-5-45-1-3-15-top-100

135 0.211289 submit-3-3-5-0-0-0-5-5-1-5-45-1-3-15-top-100-remove-hotness

136 0.213702 submit-0-0-0-0-0-0-0-0-0-1-5-1-3-15-top-100

137 0.212764 submit-1-0-0-0-0-0-1-1-0-1-5-1-3-15-top-100

138 0.213433 submit-0-0-0-0-0-0-0-0-1-2-3-1-2-3-top-100

139 0.213491 submit-0-0-0-0-0-0-0-0-1-3-25-1-3-25-top-100  # rate = 3 is much more important

140 0.213325 submit-0-0-0-0-0-0-0-0-1-3-25-1-3-25-top-600

141 0.213366 submit-3-0-0-0-0-0-3-3-1-3-15-1-3-15-top-200

142 0.213432 submit-3-3-5-0-0-0-5-5-0-0-15-15-15-1-3-15-top-200 .csv

143 0.213308 submit-3-3-5-0-0-0-5-5-0-0-15-15-15-1-3-15-top-50

144 0.214930 submit-3-3-5-0-0-0-5-5-0-0-15-15-45-1-3-15-top-50

145 0.212846 submit-3-3-5-0-0-0-5-5-0-0-15-45-15-1-3-15-top-50

146 0.214009 submit-3-3-5-0-0-0-5-5-0-0-15-15-45-1-3-15-top-100