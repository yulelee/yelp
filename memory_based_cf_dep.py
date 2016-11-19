from math import sqrt
import numpy as np
import pandas as pd
from sklearn import cross_validation as cv

file_city_name = 'las_vegas'

header = ['user_id', 'business_id', 'rating']
data = pd.read_csv('data/' + file_city_name + '_reviews_ratings_only.txt', sep='\t', names=header)

n_users, n_businesses = data.user_id.unique().shape[0], data.business_id.unique().shape[0]

training_set, test_set = cv.train_test_split(data, test_size=0.25)

# construct the user based matrix
# matrix[user_id][bus_id] = rating
def construct_matrix(dataset):
    user_matrix = {}
    for (_, user_id, business_id, rating) in dataset.itertuples():
        if user_id not in user_matrix: user_matrix[user_id] = {}
        user_matrix[user_id][business_id] = rating
    return user_matrix

train_matrix, test_matrix = construct_matrix(training_set), construct_matrix(test_set)

# return the dot product of two sparse vector, 
# assuming the second vector has more elements
def dot(x, y):
    if len(x) > len(y): return dot(y, x)
    return sum(value * y.get(key, 0) for (key, value) in x.items())

# count the nuique users in the training set
# create a mapping between the user_id and the index in the similarity matrix
num_user_train = 0
train_user_id_index = {}
for row in training_set.itertuples():
    if row[1] not in train_user_id_index: 
        train_user_id_index[row[1]] = num_user_train
        num_user_train += 1
train_user_index_id = {index: id for (id, index) in train_user_id_index.items()}

# create the vector stores the square sum of the vectors
# the index is the same stored in the train_user_index_id
# the means ratings for each user is also computed
print 'computing sum and mean'
square_sum = np.zeros(num_user_train)
mean_rating = np.zeros(num_user_train)
for i in range(num_user_train):
    user_ratings = train_matrix[train_user_index_id[i]]
    square_sum[i] = sqrt(sum(value ** 2 for value in user_ratings.values()))
    mean_rating[i] = sum(value for value in user_ratings.values()) * 1.0 / len(user_ratings)
    # with this mean, we can normalize the ratings
    for business_id in user_ratings.keys():
        user_ratings[business_id] -= mean_rating[i]


# compute the similarity matrix, which is the pairwise cosine similarity for 
# all of the users in the training set
print 'computing similarity'
user_simi = np.zeros((num_user_train, num_user_train))
for i in range(num_user_train):
    print i
    for j in range(num_user_train):
        # because the ratings has been normalized, the similarity might be negative,
        # in that case, we just set the similarity to be zero
        user_simi[i][j] = max(dot(train_matrix[train_user_index_id[i]], train_matrix[train_user_index_id[j]]) \
                / (square_sum[i] * square_sum[j]), 0)
simi_sum = np.sum(user_simi, axis = 0)

# Predcition 

# this file is deprecated, so slow...







