""" Alternating Least Square Recommendation Model
    for MovieLens data with Mean Average Precision (MAP) Calculation  """

## Imports
from __future__ import print_function
import sys
import datetime
from pyspark import SparkConf, SparkContext
from pyspark.sql import SparkSession
from pyspark.mllib.recommendation import ALS, MatrixFactorizationModel, Rating
from pyspark.mllib.evaluation import RegressionMetrics, RankingMetrics
import numpy as np


## CONSTANTS

APP_NAME = "My Spark Application"

    
## Main functionality

def main(sc, filename):
        
    #Read in the dataset as an RDD called rawdata    
    rawdata = sc.textFile(filename, 20) 
    header = rawdata.first()
    rawdata = rawdata.filter(lambda x: x!= header)
    rawdata.take(3)
    num_data = rawdata.count()
        
    #We only want the first 3 columns
    rawratings= rawdata.map(lambda x: x.split(',')).map(lambda x: (x[0], x[1], x[2]))
    rawratings.take(3)

    #Convert the data to an RDD of type "Ratings"
    ratings = rawratings.map(lambda x: Rating(int(x[0]), int(x[1]), float(x[2])))
    ratings.cache()
    num_data=ratings.count()
    print(num_data)
    
    # Creating training and testing sets to create model and compute RMSE
    train_data, test_data = ratings.randomSplit([0.8, 0.2])
    train_data.cache()
    test_data.cache()
    tr = train_data.count()
    te = test_data.count()
    print(tr, te)
    ratings.unpersist()
    
    #Using optimal parameters obtained from K-fold cross validation with grid search
    iterations = 10
    rank = 75
    regParam = 0.1
    model=ALS.train(train_data, rank=rank, iterations=iterations, lambda_=regParam, blocks=-1, nonnegative=False, seed=None)

    #Evaluate the model on test data, and compute RMSE and MSE    
    usersProducts = test_data.map(lambda p: (p[0], p[1]))
    usersProducts.count()
    predictions = model.predictAll(usersProducts).map(lambda r: ((r[0], r[1]), r[2]))
    ratingsAndPredictions = test_data.map(lambda r: ((r[0], r[1]), r[2])).join(predictions)
    predictedAndTrue = ratingsAndPredictions.map(lambda d: (d[1][0], d[1][1]))
    regressionMetrics = RegressionMetrics(predictedAndTrue)

    print("Mean Squared Error = " + str(regressionMetrics.meanSquaredError))
    print("Root Mean Squared Error = " + str(regressionMetrics.rootMeanSquaredError))
    MSE = regressionMetrics.meanSquaredError
    RMSE = regressionMetrics.rootMeanSquaredError
    

    # RECORD START TIME
    timestart2 = datetime.datetime.now()
    
    ################################################################
    ####        Computation of Mean Average Precision           ####
    ################################################################
    
    #Extract item factors matrix from Matrix Factorization Model
    itemFactors = model.productFeatures().map(lambda x: (x[1])).collect()
    print(len(itemFactors), len(itemFactors[0]))
    itemMatrix = np.array(itemFactors)
    print(itemMatrix[1])
    #Define item factor matrix as a broadcast variable    
    imBroadcast = sc.broadcast(itemMatrix)    

    #Obtain the recommendations for each user   
    allRecs = model.userFeatures().map(lambda m: (m[0], get_recs(m[1])))
    allRecs.cache()
    #Obtain the actual ratings for each user
    userMovies = ratings.map(lambda x: (x[0],x[1])).groupByKey().mapValues(list).sortBy(lambda k: k[0])
    userMovies.cache()
    
    #Join the actual and predicted movie lists
    predictedAndTrueForRanking = allRecs.join(userMovies).map(lambda s: (np.array(s[1][0]), np.array(s[1][1])))
    #Apply the MAP computation on the predicted and actual movie lists
    #Since the recommended movies have been sorted in descending order, the MAP is a measure of
    # how relevant the recommendations are and that the most relevant also show up high in the results.
    rankingMetrics = RankingMetrics(predictedAndTrueForRanking)
    print("Mean Average Precision = " + str(rankingMetrics.meanAveragePrecision))
    MAP = rankingMetrics.meanAveragePrecision

 
    # PRINT ELAPSED TIME    
    timeend2 = datetime.datetime.now()
    timedelta2 = round((timeend2-timestart2).total_seconds(), 2) 
    print ("Time taken to compute mean average precision: " + str(timedelta2) + " seconds")     

    output = sc.parallelize([('ALS', MSE, RMSE, MAP, timedelta2)])
    output.saveAsTextFile('/usr/local/spark/als_test_output')    
    sc.stop()
  

##OTHER FUNCTIONS/CLASSES

def get_recs(user_factor_vector):
    userVector = np.array(user_factor_vector)
    #Compute the dot product of the user vector with the item factor matrix to obtain ratings
    scores = imBroadcast.value.dot(userVector)
    #Apply an index value to each row value and proceed to sort the ratings in descending order.
    #This preserves the corresponding rating to each movieID
    sortedWithId = sc.parallelize(scores).zipWithIndex().sortBy(lambda x: x[0], ascending = False)
    print(sortedWithId.take(10))
    #Add 1 to the index value because the indexes are zero-based but the movieId's are 1-based
    recommendedIds=sortedWithId.map(lambda x: (x[1]+1))
    #The final result is the list of movieId's for the recommended movies, sorted from from highest to lowest
    #The actual ratings values are discarded
    return recommendedIds


if __name__ == "__main__":
    # Configure OPTIONS
    conf = SparkConf().setAppName(APP_NAME)
    conf = conf.setMaster("local[*]")
    #in cluster this will be like
    #"spark://ec2-0-17-03-078.compute-#1.amazonaws.com:7077"
    sc   = SparkContext(conf=conf)
    spark = SparkSession(sc)
    filename = sys.argv[1]
    # Execute Main functionality
    main(sc, filename)



