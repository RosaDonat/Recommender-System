""" Alternating Least Square Recommendation Model
    for MovieLens data with Top 10 Recommendations for User  """

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
    ratings.take(3)
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
    model.userFeatures().count()
    model.productFeatures().count()
    model.rank
    model.userFeatures
    
    #Evaluate model on teswt data and obtain MSE and RMSE
    usersProducts = test_data.map(lambda p: (p[0], p[1]))
    usersProducts.count()
    predictions = model.predictAll(usersProducts).map(lambda r: ((r[0], r[1]), r[2]))
    ratingsAndPredictions = test_data.map(lambda r: ((r[0], r[1]), r[2])).join(predictions)
    predictedAndTrue = ratingsAndPredictions.map(lambda d: (d[1][0], d[1][1]))
    regressionMetrics = RegressionMetrics(predictedAndTrue)
    ratingsAndPredictions.count()
    predictedAndTrue.count()
    print("Mean Squared Error = " + str(regressionMetrics.meanSquaredError))
    print("Root Mean Squared Error = " + str(regressionMetrics.rootMeanSquaredError))
    MSE = regressionMetrics.meanSquaredError
    RMSE = regressionMetrics.rootMeanSquaredError
   

    # RECORD START TIME
    timestart = datetime.datetime.now()
    
    ##############################################
    ####        Make a few predictions       ####
    ##############################################
    
    #Read in the movie ID's and titles
    movies = sc.textFile('/usr/local/spark/movies.csv')
    header = movies.first()
    movies = movies.filter(lambda x: x!= header)
    titles = movies.map(lambda x: x.split(',')).map(lambda s: (int(s[0]), s[1])).collectAsMap()
        
    # As an example, predict a rating for item 123 for user 1
    temp = sc.parallelize([(1, 123)])
    predictedRating = model.predictAll(temp).collect()
    print(titles[123])
    print(predictedRating)
    
    #Predict the top 10 movies for a user (me)
    #My ratings have been appended to ratings.csv input file
    userId = 200000
        
    #Find out how many movies user has rated
    moviesForUser = ratings.map(lambda x: ((x[0]),(x[1],x[2]))).lookup(userId)
    print(len(moviesForUser))
    print(moviesForUser[0])
    #Find the top 10 rated movies for user 
    moviesForUser = sc.parallelize(moviesForUser)
    top10Rated = moviesForUser.sortBy(lambda s: s[1], ascending = False).take(10)
    print(top10Rated[0])
    print("The following are the top 10 rated movies by this user: \n")
    for item in top10Rated:
        print(str(titles[item[0]]) + (', ') + str(item[1]) + ('\n'))
    
    #Now look at the top 10 recommendations for this user
    #Extracting the movie title to show recommendations
    K = 10
    topKRecs = model.recommendProducts(userId, K)
    print("The following are the top 10 recommended movies for this user: \n")
    for item in topKRecs:
        print(str(titles[item[1]]) + (', ') + str(item[2]) + ('\n'))

    # PRINT ELAPSED TIME    
    timeend = datetime.datetime.now()
    timedelta = round((timeend-timestart).total_seconds(), 2) 
    print ("Time taken to execute recommendations: " + str(timedelta) + " seconds")     
    
    sc.stop()
  

##OTHER FUNCTIONS/CLASSES



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



