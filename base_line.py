# base line model predict the rating of an user to a restaurant just using
# the average rating of this user's history, cannot make recommendations

import scipy.sparse as sp
import numpy as np
from math import sqrt
from sklearn.metrics import mean_squared_error
import load_matrix
import utils

train_data_matrix = load_matrix.train_data_matrix
test_data_matrix = load_matrix.test_data_matrix
all_predictions = np.zeros((load_matrix.n_users, load_matrix.n_items))

for user_idx in range(load_matrix.n_users):
	nonzero_index = np.nonzero(load_matrix.train_data_matrix[user_idx,])[0]
	nonzero_values = load_matrix.train_data_matrix[user_idx, nonzero_index]
	prediction = nonzero_values.mean()
	all_predictions[user_idx, nonzero_index] = prediction

print 'rmse:', utils.rmse(all_predictions, load_matrix.test_data_matrix)