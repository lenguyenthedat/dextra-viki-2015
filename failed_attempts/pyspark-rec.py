from pyspark.mllib.recommendation import ALS, MatrixFactorizationModel, Rating

complete_ratings_file = "data/behavior-ml-score-ints.csv"
complete_ratings_raw_data = sc.textFile(complete_ratings_file)


# Parse
complete_ratings_data = complete_ratings_raw_data.map(lambda line: line.split(",")).map(lambda tokens: (int(tokens[0]),int(tokens[1]),float(tokens[2]))).cache()
    
print "There are %s ratings in the complete dataset" % (complete_ratings_data.count())

training_RDD, test_RDD = complete_ratings_data.randomSplit([7, 3], seed=0L)

complete_model = ALS.train(training_RDD, best_rank, seed=seed, 
                           iterations=iterations, lambda_=regularization_parameter)

test_for_predict_RDD = test_RDD.map(lambda x: (x[0], x[1]))

predictions = complete_model.predictAll(test_for_predict_RDD).map(lambda r: ((r[0], r[1]), r[2]))
rates_and_preds = test_RDD.map(lambda r: ((int(r[0]), int(r[1])), float(r[2]))).join(predictions)
error = math.sqrt(rates_and_preds.map(lambda r: (r[1][0] - r[1][1])**2).mean())
    
print 'For testing data the RMSE is %s' % (error)


#
new_user_unrated_movies_RDD = (complete_movies_data.filter(lambda x: x[1]!=new_user_ID)
                      .map(lambda x: (new_user_ID, x[0])))
                      
complete_model.predictAll(new_user_unrated_movies_RDD)




# http://spark.apache.org/docs/latest/mllib-collaborative-filtering.html

from pyspark.mllib.recommendation import ALS, MatrixFactorizationModel, Rating

# Load and parse the data
data = sc.textFile("data/behavior-ml-score-ints.csv")
ratings = data.map(lambda l: l.split(',')).map(lambda l: Rating(int(l[0]), int(l[1]), float(l[2])))

training_RDD, test_RDD = ratings.randomSplit([7, 3], seed=0L)

# Build the recommendation model using Alternating Least Squares
rank = 10
numIterations = 20
model = ALS.train(training_RDD, rank, numIterations)

# Evaluate the model on training data
testdata = test_RDD.map(lambda p: (p[0], p[1]))
predictions = model.predictAll(testdata).map(lambda r: ((r[0], r[1]), r[2]))
ratesAndPreds = ratings.map(lambda r: ((r[0], r[1]), r[2])).join(predictions)
MSE = ratesAndPreds.map(lambda r: (r[1][0] - r[1][1])**2).mean()
print("Mean Squared Error = " + str(MSE))

# Save and load model
model.save(sc, "myModelPath")
sameModel = MatrixFactorizationModel.load(sc, "myModelPath")


