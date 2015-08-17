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

# Collaborative Filtering (Jaccard Index) plus feature similarity

    $ python viki-videos-similarity.py # Pre-procesing `#videos x #videos` matrix, roughly 1.5 hours

Then run recommender with either one of these

    $ python viki-users-recommender.py # batch process - 20 mins

This is more practical since `#videos x #videos` matrix is much smaller.
Weights can be set manually:

    top_videos_limit = 20
    sim_features = ['sim_country', 'sim_language', 'sim_adult',
                    'sim_content_owner_id', 'sim_broadcast', 'sim_episode_count',
                    'sim_genres', 'sim_cast',
                    'jaccard_1_3', 'jaccard_2_3', 'jaccard_3_3']
    weight_features = [0,0,0,
                       0,0,0,
                       0,0,
                       0,0,0]
    weight_scores = [0,0,0]

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
53- 00-00-00-00-00-00-00-00-01-03-05-01-02-03 Result: 0.189109 (top 100 only) - factor in hotness properly
54- 00-00-00-00-00-00-00-00-01-03-05-01-02-03 Result: 0.190801 (top 20 only)
55- 00-00-00-00-00-00-00-00-01-03-05-01-02-03 Result: 0.192084 (top 10 only)
56- 01-01-01-00-00-00-01-01-00-01-03-00-01-03 Result: 0.191566
57- new hot videos                            Result: 0.201594
58- new hot videos (only count score= 2 or 3) Result: 0.202451
59- new hot videos (score= 2 or 3, sqrt)      Result: 0.181191
60- new hot videos (score= 2 or 3, pow 2)     Result: 0.189585
61- 01-01-01-00-00-00-01-01-00-04-07-00-01-02-top-20  Result: 0.199957
62- 03-03-05-00-00-00-05-05-00-02-10-01-02-03-top-20  Result: 0.199543
63- 03-03-05-00-00-00-05-05-00-02-10-01-02-03-top-20  Result: 0.202127 - freshness
64- 03-03-05-00-00-00-05-05-00-02-10-01-02-03-top-10  Result: 0.204343
65- 03-03-05-00-00-00-05-05-00-02-10-01-02-03-top-100 Result: 0.202351
66- 03-30-05-00-00-00-05-05-01-05-25-01-05-25-top-10  Result: 0.206155
67- 03-03-05-00-00-00-05-15-01-15-45-01-05-45-top-10  Result: 0.204596
68- 03-03-05-00-00-00-05-05-01-05-45-01-05-15-top-10  Result: 0.206163
69- 03-03-05-00-00-00-05-05-00-05-45-00-05-15-top-100 Result: 0.201184 - hotness^2
70- 03-03-05-00-00-00-05-05-00-05-05-00-05-15-top-50  Result: 0.195498
71- 00-00-00-00-00-00-00-00-00-15-45-00-05-15-top-50  Result: 0.201141
72- 00-00-00-00-00-00-00-00-00-15-45-00-05-15-top-10  Result: 0.204553 Minmax
73- 03-03-05-00-00-00-05-05-00-05-45-00-05-15-top-100 Result: 0.201493 - minmax freshness 1-2
74- 03-03-05-00-00-00-05-05-01-05-45-01-05-15-top-10  Result: 0.203345 No freshness
75- 03-03-05-00-00-00-05-05-01-05-45-01-05-15-top-10  Result: 0.205918 freshness^2
76- 03-03-05-00-00-00-05-05-01-05-45-01-05-15-top-10  Result: 0.205935 freshness^2 minmax 1-2
77- 03-03-05-00-00-00-05-05-01-05-45-01-05-15-top-10  Result: 0.168555 amazon-limit-by-top-perf
78- 03-03-05-00-00-00-05-05-01-05-45-01-05-15-top-10  Result: 0.184913 limit -by-hotness-combine-by-aws-perf
79- 03-03-05-00-00-00-05-05-01-05-45-01-05-15-top-10  Result: 0.202482 hotness only when score = 3
80- 3-3-0-0-0-0-5-25-1-15-45-1-15-45-top-10 Result: 0.203116
81- 3-3-25-0-0-0-5-1-1-15-45-1-15-45-top-5 Result: 0.204379
82- 3-3-25-0-0-0-5-25-1-15-45-1-15-45-top-5-nofresh-nohotness Result: 0.199439
83- 3-3-25-0-0-0-5-25-1-15-45-1-15-45-top-25-nohotness Result: 0.184235
84- 90-5-5-0-0-0-5-15-1-10-45-1-5-7-top-100 Result: 0.202711-hot^2
85- 99-5-5-0-0-0-5-15-1-45-45-1-5-5-top-25 Result: 0.202381
86- 99-5-5-0-0-0-5-15-1-45-45-1-5-5-top-100 Result: 0.168672-hotnessminmax1-2-newjaccard
87- 99-5-5-0-0-0-5-15-1-45-45-1-5-5-top-100 Result: 0.180541-hotnessminmax0-1-newjaccard
88- 99-5-5-0-0-0-5-15-1-45-45-1-5-5-top-100 Result: 0.188778-noscale-newjaccard
89- 3-3-5-0-0-0-5-5-1-5-45-1-5-15-top-10 Result: 0.194169-newjaccard is bad
90- 3-3-5-0-0-0-5-5-1-5-45-1-5-15-top-10 Result: 0.206163
91- 3-3-25-0-0-0-5-0-1-25-45-1-5-5-top-10 Result: 0.205861
92- 25-5-25-0-0-0-15-15-0-0-45-0-0-5-top-10 Result: 0.199735
93- 25-5-25-0-0-0-0-0-0-45-45-0-5-5-top-10 Result: 0.204245
94- 03-03-05-00-00-00-05-05-01-05-45-01-05-15-top-10  Result: 0.211681 mimic #68 with a split for Hotness (male / female / overall)
95- 00-00-00-00-00-00-00-00-00-00-00-01-05-15-top-10  Result: 0.201120 hotness (split) only
96- 03-03-05-00-00-00-05-05-01-05-45-01-05-15-top-10  Result: 0.210312 mimic #94 with hotness^2
97- 03-03-05-00-00-00-05-05-01-05-45-01-05-15-top-25  Result: 0.211086 mimic #94 with top25
98- 03-03-05-00-00-00-05-05-01-05-45-01-05-15-top-10  Result: 0.212555 best_videos to be divided (f,m,o)
99- 03-03-05-00-00-00-05-05-01-25-45-01-02-03-top-10  Result: 0.212485
100 03-03-05-00-00-00-05-05-01-05-45-01-05-15-top-10  Result: 0.212396 #mimic 98 but treat hot_overall as female
101 03-03-05-00-00-00-05-05-01-05-45-01-05-15-top-10  Result: 0.214330 #fresh^2
102 03-03-05-00-00-00-05-05-01-05-45-01-05-15-top-10  Result: 0.213760 #fresh^4
103 15-03-05-00-00-00-15-15-01-05-45-01-05-15-top-15  Result: 0.212328.csv
104 03-03-05-00-00-00-05-05-01-05-45-01-05-15-top-10  Result: 0.214434 #fresh^2, updated hot_o
105 03-03-05-00-00-00-05-05-01-05-45-01-05-15-top-10  Result: 0.213791 bestness
106 03-03-05-00-00-00-05-05-01-05-45-01-05-15-top-10  Result: 0.189606 bestness no hotness
107 03-03-05-00-00-00-05-05-01-05-45-01-05-15-top-10  Result: bestness (with top 10 instead of 5 per movie)
108 03-03-05-00-00-00-05-05-15-15-45-01-02-03-top-10  Result: 0.214066
109 03-03-05-00-00-00-05-05-25-25-25-01-05-15-top-10  Result: 0.214225 fresh^3
110 03-03-05-00-00-00-05-05-01-05-45-01-05-15-top-10  Result: 0.214307 fresh^3
111 svd-v1 Result: 0.194942
112 submit-3-3-5-0-0-0-5-5-1-5-45-95-1-5-15-top-10-0.214213
113 submit-3-3-5-0-0-0-5-5-1-5-25-25-1-2-3-top-10-0.21359
114 submit-0-0-0-0-0-0-0-0-0-1-1-1-0-1-1-top-10-0.212152
115 submit-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-top-10-0.205756
116 submit-3-3-5-0-0-0-5-5-1-5-45-25-1-5-15-top-10-0.214295
117 submit-5-3-0-0-0-0-5-15-1-10-25-25-0-1-5-top-15-0.208086 no punishment if score less than 0, fixed jaccard to exclude not-hot movies
118 submit-3-3-5-0-0-0-5-5-1-5-45-95-1-5-15-top-10-0.214475 # old params
119 submit-3-3-5-0-0-0-5-5-1-5-45-0-1-5-15-top-10-0.21427
120 submit-3-3-5-0-0-0-5-5-1-5-95-145-1-5-15-top-10-0.214474
121 submit-3-3-5-0-0-0-5-5-1-5-45-95-1-5-15-top-10-0.212959
122 submit-3-3-5-0-0-0-5-5-1-5-25-25-1-5-25-top-10-0.214205
123 submit-3-3-5-0-0-0-5-5-1-5-45-95-1-5-15-top-10-minmax-0.209646
124 submit-3-3-5-0-0-0-5-5-1-5-15-15-1-5-15-top-10-minmax-0.212232
125 submit-3-3-5-0-0-0-5-5-1-5-10-10-1-5-15-top-10-minmax-0.209993
126 submit-3-3-5-0-0-0-5-5-1-5-25-25-1-5-15-top-10-minmax-0.21306
127 submit-3-3-5-0-0-0-5-5-1-5-45-1-5-15-top-10-no-svd-0.214074
128 submit-3-3-5-0-0-0-5-5-1-5-15-1-5-15-top-100-0.212358 - filter before scaling.
129 submit-3-3-5-0-0-0-5-5-1-5-15-1-5-15-top-10-0.204905
130 submit-3-3-5-0-0-0-5-5-1-5-25-1-5-15-top-50-0.212648
131 submit-3-3-5-0-0-0-5-5-1-5-10-1-3-5-top-25-0.204847
132 submit-3-3-5-0-0-0-5-5-1-5-45-1-3-15-top-25-0.212818
133 submit-3-3-5-0-0-0-5-5-1-5-45-1-3-15-top-50-0.213767
134 submit-3-3-5-0-0-0-5-5-1-5-45-1-3-15-top-100-0.213771
135 submit-3-3-5-0-0-0-5-5-1-5-45-1-3-15-top-100-remove-hotness-0.211289
136 submit-0-0-0-0-0-0-0-0-0-1-5-1-3-15-top-100-0.213702
137 submit-1-0-0-0-0-0-1-1-0-1-5-1-3-15-top-100-0.212764
138 submit-0-0-0-0-0-0-0-0-1-2-3-1-2-3-top-100-0.213433
139 submit-0-0-0-0-0-0-0-0-1-3-25-1-3-25-top-100-0.213491 # rate = 3 is much more important
140 submit-0-0-0-0-0-0-0-0-1-3-25-1-3-25-top-600-0.213325
141 submit-3-0-0-0-0-0-3-3-1-3-15-1-3-15-top-200-0.213366

The reason why HOT VIDEOS dominated CF is because viki's homepage currently dominated by:
- Top banner
- Popular show
- Top Drama
- # Gender filter for male / female

TODO:
- treat videos user watched differently based on the time he / she watched it
- utilize ratio instead of score
- limit top videos for sim calculatino before scaling