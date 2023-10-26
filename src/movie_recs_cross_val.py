""" Alternating Least Square Recommendation Model
    for MovieLens data with K-fold cross validation  """

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
    # RECORD START TIME
    timestart = datetime.datetime.now()
    
    #Read in the dataset as an RDD called rawdata    
    
    rawdata = sc.textFile(filename, 20) 
    header = rawdata.first()
    rawdata = rawdata.filter(lambda x: x!= header)
    rawdata.take(3)
    num_data = rawdata.count()
    print(num_data)
    
    #We only want the first 3 columns
    rawratings= rawdata.map(lambda x: x.split(',')).map(lambda x: (x[0], x[1], x[2]))
    rawratings.take(3)

    #Convert the data to an RDD of type "Ratings"
    ratings = rawratings.map(lambda x: Rating(int(x[0]), int(x[1]), float(x[2])))
    ratings.take(3)
    ratings.cache()
    num_data=ratings.count()
    print(num_data)
    
    ## Tuning model parameters

    # Creating training and testing sets to evaluate parameters
    train_data, test_data = ratings.randomSplit([0.8, 0.2])
    train_data.cache()
    test_data.cache()
    tr = train_data.count()
    te = test_data.count()
    print(tr, te)
    ratings.unpersist()
    
    # The parameter settings for the AlS model with regularization parameter:
    rank = [5, 10, 20, 50, 75]
    iterations = [10]
    reg_param = [0.001, .01, .1, 1.0, 10.0]
    k = 10
    partitions = train_data.randomSplit([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
    
    #Start outer loop in which each parameter set is applied to the linear model (depends on number of parameters defined above) 
    for s in range(len(rank)):
        for r in range(len(iterations)):
            for m in range(len(reg_param)):                
                rnk = rank[s]
                itr = iterations [r]
                rp = reg_param[m]
                                
                #Start inner loop in which k-fold cross validation is applied (k times)
                metrics = []    
                for i in range(k):
                    for j in range(k):
                        if i != j:            
                            try:
                                trainingSet = trainingSet.union(partitions[j])
                            except:
                                trainingSet = partitions[j]
                        testSet = partitions[i]
                    print("New partitions created: ", trainingSet.count(), testSet.count())
                    metric = evaluate(trainingSet, testSet, rnk, itr, rp)
                    metrics.append(metric)
                    del trainingSet
                    del testSet
                print(metrics)
                mean_error = sum(metrics)/k
                try:
                    errors.append(mean_error)
                except:
                    errors = []
                    errors.append(mean_error)
                print('mean error = ', mean_error)
                param_set = sc.parallelize([(mean_error, rnk, itr, rp)])
                try:
                    param_grid = param_grid.union(param_set)
                except:
                    param_grid = param_set
                            
                            
    print("Errors from k-fold cross validation (30 parameter sets using k=5): ", errors)                     
    min_error = min(errors)
    print("minimum error obtained: ", min_error)
    
    #Find the parameter set that produced smallest error
    result = param_grid.filter(lambda keyvalue: keyvalue[0] == min_error).flatMap(lambda x: x).collect()
    print(result)
    rnk = result[1]
    itr = result[2]
    rp = float(result[3])
        
    # PRINT ELAPSED TIME    
    timeend = datetime.datetime.now()
    timedelta = round((timeend-timestart).total_seconds(), 2) 
    print ("Time taken to execute cross validation: " + str(timedelta) + " seconds")     
    
    # RECORD START TIME
    timestart2 = datetime.datetime.now()
    
    #Final run with optimal parameters
    final_metric = evaluate(train_data, test_data, rnk, itr, rp)
    print('Final RMSE: ', final_metric, 'Final MSE: ', final_metric**2 )

    # PRINT ELAPSED TIME    
    timeend2 = datetime.datetime.now()
    timedelta2 = round((timeend2-timestart2).total_seconds(), 2) 
    print ("Time taken to execute final model train and test: " + str(timedelta2) + " seconds")     

    output = sc.parallelize([('ALS', errors, result, final_metric, final_metric**2, timedelta, timedelta2)])
    output.saveAsTextFile('/usr/local/spark/als_output')    
    sc.stop()
  

##OTHER FUNCTIONS/CLASSES
  
def evaluate(train, test, rank, iterations, regParam):
    model = ALS.train(train, rank=rank, iterations=iterations, lambda_ = regParam, blocks=-1, nonnegative=False, seed=None)
    usersProducts = test.map(lambda p: (p[0], p[1]))
    print(usersProducts.count())
    predictions = model.predictAll(usersProducts).map(lambda r: ((r[0], r[1]), r[2]))
    ratingsAndPredictions = test.map(lambda r: ((r[0], r[1]), r[2])).join(predictions)
    predictedAndTrue = ratingsAndPredictions.map(lambda d: (d[1][0], d[1][1]))
    regressionMetrics = RegressionMetrics(predictedAndTrue)
    ratingsAndPredictions.count()
    predictedAndTrue.count()
    return regressionMetrics.rootMeanSquaredError
    


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



