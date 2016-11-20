# base line model predict the rating of an user to a restaurant just using
# the average rating of this user's history, cannot make recommendations

import scipy.sparse as sp
import numpy as np
from math import sqrt
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
import load_matrix
import utils

train_data_matrix = load_matrix.train_data_matrix
test_data_matrix = load_matrix.test_data_matrix

############################################################
# mean base line

test_predictions = np.zeros((load_matrix.n_users, load_matrix.n_items))
train_predictions = np.zeros((load_matrix.n_users, load_matrix.n_items))

for user_idx in range(load_matrix.n_users):
	nonzero_index = np.nonzero(load_matrix.train_data_matrix[user_idx,])[0]
	nonzero_values = load_matrix.train_data_matrix[user_idx, nonzero_index]
	prediction = nonzero_values.mean()

	train_predictions[user_idx, nonzero_index] = prediction

	test_nonzero_index = np.nonzero(load_matrix.test_data_matrix[user_idx,])[0]
	test_predictions[user_idx, test_nonzero_index] = prediction

print 'mean training rmse:', utils.rmse(train_predictions, load_matrix.train_data_matrix)
print 'mean testing rmse:', utils.rmse(test_predictions, load_matrix.test_data_matrix)

############################################################
# linear regression baseline

m = train_data_matrix + test_data_matrix
user_mean_rating = m.sum(1) / (m != 0).sum(1)
restaurant_mean_rating = m.sum(0) / (m != 0).sum(0)

def extract_features(dataset):
	nonzero_positions = np.nonzero(dataset)
	data_points = np.zeros((len(nonzero_positions[0]),2))
	data_points[:,0] = user_mean_rating[nonzero_positions[0]]
	data_points[:,1] = restaurant_mean_rating[nonzero_positions[1]]

	ratings = dataset[np.nonzero(dataset)]
	ratings = ratings.reshape(len(ratings), 1)
	return (data_points, ratings)

(features, ratings) = extract_features(train_data_matrix)
(test_features, test_ratings) = extract_features(test_data_matrix)

lr = LinearRegression()
lr.fit(features, ratings)

train_predictions = lr.predict(features)
test_predictions = lr.predict(test_features)

print 'linear regression training rmse:', utils.list_rmse(train_predictions, ratings)
print 'linear regression testing rmse:', utils.list_rmse(test_predictions, test_ratings)

############################################################
# average of average base line

print 'average of average training rmse:', utils.list_rmse(features.mean(axis=1), ratings)
print 'average of average testing rmse:', utils.list_rmse(test_features.mean(axis=1), test_ratings)


