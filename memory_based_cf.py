from math import sqrt
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.metrics import mean_squared_error
import load_matrix
import utils

file_city_name = utils.file_city_name

train_data_matrix = load_matrix.train_data_matrix
test_data_matrix = load_matrix.test_data_matrix

print 'computing similarity'

user_similarity = pairwise_distances(train_data_matrix, metric='cosine')
item_similarity = pairwise_distances(train_data_matrix.T, metric='cosine')

# fix the negative similarities
user_similarity = user_similarity + np.abs(np.min(user_similarity))
# item_similarity = item_similarity + np.abs(np.min(item_similarity))

def predict(ratings, similarity, selector, type='user'):
    if type == 'user':
    	selector = selector + 0.000000000001
        # mean_user_rating = ratings.mean(axis=1)
        # #You use np.newaxis so that mean_user_rating has same format as ratings
        # ratings_diff = (ratings - mean_user_rating[:, np.newaxis]) 
        # pred = mean_user_rating[:, np.newaxis] + similarity.dot(ratings_diff) / similarity.dot(selector)
       
        a = similarity.dot(ratings) 
        b = similarity.dot(selector)
        
        # print np.max(a), np.max(b), np.min(a), np.min(b)
        pred = a / b
    elif type == 'item':
        pred = ratings.dot(similarity) / selector.dot(similarity)
    return pred

print 'making predictions'

# matrix contains some 1s to select elements
select = np.copy(train_data_matrix) 
select[np.where(train_data_matrix != 0)] = 1

user_prediction = predict(train_data_matrix, user_similarity, select, type='user')
item_prediction = predict(train_data_matrix, item_similarity, select, type='item')

# print np.max(item_prediction), np.min(item_prediction), np.max(user_prediction), np.min(user_prediction)

print 'User-based CF RMSE: ' + str(utils.rmse(user_prediction, test_data_matrix))
print 'Item-based CF RMSE: ' + str(utils.rmse(item_prediction, test_data_matrix))


with open('data/' + file_city_name + 'memory_based_cf_user' + '_all_predictions.np', 'w') as file:
	np.save(file, user_prediction)

with open('data/' + file_city_name + 'memory_based_cf_item' + '_all_predictions.np', 'w') as file:
	np.save(file, item_prediction)
