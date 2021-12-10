# Recommendation system using collaborative filtering matrix factorization neural network

## Introduction
A  recommendation system is an information filtering system that seeks to predict the "rating" or "preference" a user would give to an item, thus generating recommendations for the user. Recommendation systems are of two types  
1. Content based filtering (only based on the current users past data)
2. Collaborative Filtering (finding similar users & finding similar products)  
 (i). The rating given to the item by “similar” users (user-based CF)  
 (ii). The rating given to “similar” items by the user (item-based CF)  
 
The problem with these is that user-item matrices are large and sparse. Instead, the industry standard is to find latent features by reducing the matrix with matrix factorization.

## About the problem & Methodology

The dataset contains around 1,75,00,000 rows of data containing listening behavior of 360,000 users in a music streaming app. The columns contain users, artists they listened to, the number of plays for each artist. By evaluating this data, we are supposed to find recommendations for the user.  

2 models are used:


1. Multi layer perceptron model - A deep neural network is used to learn the non-linear function rather than a linear one and in doing so utilize the expressiveness of the final model which will take into account complex relationship between datapoints  
2. The Matrix Factorization network is used which is a regular point-wise matrix factorization to approximate the factorization of a large and sparse (user x item) matrix into the two lower dimensional matrices (user x features) and (features x items) using embedding layer. 
3. Combine both models to utilize the strength of both models to make better predictions  

## Improvement
Instead of our very simple matrix factorization function implemented here, we could use an ALS or BPR model to factor our matrices.

1. ALS - Alternating least squares - It is an iterative optimization process used to segregate the matrix ito two matrices with user * features  and artists * features  -> thus finding latent/hidden features. For every iteration, we try to arrive closer and closer to a factorized representation of our original data R = U * V. We iteratively alternate between optimizing U and fixing V and vice versa. The key insight is that you can turn the non-convex optimization problem(no global minimum) into an "easy" quadratic problem if you fix either U  or  V. ALS fixes each one of those alternatively. When one is fixed, the other one is computed, and vice versa.  
2. BPR - Bayesian Personalized Ranking - how to treat data that user has not seen? It may be that user loves that song but score is zero only because they have not discovered it yet. So we cant assign score of 0 , we assign scores based on baysian probablity ie. based on confidence scores. We assign a small score instead of 0 for items which the user has not interacted with, and a higher score for items which we know the user likes.
 
