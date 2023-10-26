# Recommender System using ALS on MovieLens dataset

**Objective:** 

Build a recommender system in Spark as a Matrix Factorization Model using Alternating Least Squares (ALS) to optimize the loss function. 

To begin with, the data we have is movie ratings by over 100,000 users and 27,000 movies. If we were to 
build a ratings matrix with users as rows and movies as columns, it would be an incredibly large and very 
sparse matrix. A lower dimension approximation to our user-item matrix would produce two matrices: 
one for user factors and one for item factors. 

Alternating Least Squares (ALS) is an optimization technique to solve matrix factorization problems, and 
it works by iteratively solving a series of least squares regression problems. In each iteration one of the 
matrices (user factor matrix or item factor matrix) is treated as fixed while the other is updated, and on 
the next step the updated one is treated as fixed. This continues until convergence or for a certain 
number of iterations. 

Using these matrices, we will build the recommender system using Collaborative Filtering. In this 
technique we assume that users that have exhibited similar preferences will be similar in terms of taste. 
So, to make recommendations to a user for items they have not reviewed, we use the known 
preferences of users that have exhibited similar behavior. 

The features we need to use as inputs are user ID's, movie ID's, and the ratings for each. These data will 
be read in as an RDD and then converted to a type Rating, which is a wrapper around the data [user, 
product, rating], and this data structure will be the input to the ALS model train functio
