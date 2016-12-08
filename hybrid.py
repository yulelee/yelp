# Run linear regression on all of the results

import utils
import load_matrix
import math
import numpy as np
from sklearn import linear_model

file_city_name = utils.file_city_name

train_data_matrix = load_matrix.train_data_matrix
test_data_matrix = load_matrix.test_data_matrix

print 'Loading results from models...'

# Those models' results are stored together in one matrix

memory_based_cf_user_all = np.load('data/' + file_city_name + 'memory_based_cf_user' + '_all_predictions.np')
memory_based_cf_item_all = np.load('data/' + file_city_name + 'memory_based_cf_item' + '_all_predictions.np')
sgd_all = np.load('data/' + file_city_name + 'sgd' + '_all_predictions.np')
svd_all = np.load('data/' + file_city_name + 'svd' + '_all_predictions.np')

# Split the stored results into training and testing parts

# memory_based_cf_user_train = np.copy(memory_based_cf_user_all)
# memory_based_cf_user_train[np.where(train_data_matrix == 0)] = 0
# memory_based_cf_user_test = np.copy(memory_based_cf_user_all)
# memory_based_cf_user_test[np.where(test_data_matrix == 0)] = 0

# memory_based_cf_item_train = np.copy(memory_based_cf_item_all)
# memory_based_cf_item_train[np.where(train_data_matrix == 0)] = 0
# memory_based_cf_item_test = np.copy(memory_based_cf_item_all)
# memory_based_cf_item_test[np.where(test_data_matrix == 0)] = 0

# sgd_train = np.copy(sgd_all)
# sgd_train[np.where(train_data_matrix == 0)] = 0
# sgd_test = np.copy(sgd_all)
# sgd_test[np.where(test_data_matrix == 0)] = 0

# svd_train = np.copy(sgd_all)
# svd_train[np.where(train_data_matrix == 0)] = 0
# svd_test = np.copy(svd_all)
# svd_test[np.where(test_data_matrix == 0)] = 0


# These models' results are stored separately for training and testing
# so we don't need to split them here

sorting_method = '_user_preference_cos.np'
tbr_preference_train = np.load('data/' + file_city_name + sorting_method + '_train_predictions.np')
tbr_preference_test = np.load('data/' + file_city_name + sorting_method + '_test_predictions.np')

sorting_method = '_cosine_similarities.np'
tbr_similarity_train = np.load('data/' + file_city_name + sorting_method + '_train_predictions.np')
tbr_similarity_test = np.load('data/' + file_city_name + sorting_method + '_test_predictions.np')


print 'Constructing the features and labels...'

# construct the features and labels for the linear regression
hybrid_train_feature = np.column_stack((memory_based_cf_user_all[train_data_matrix.nonzero()], \
										memory_based_cf_item_all[train_data_matrix.nonzero()], \
										sgd_all[train_data_matrix.nonzero()], \
										svd_all[train_data_matrix.nonzero()], \
										tbr_preference_train[train_data_matrix.nonzero()], 
										tbr_similarity_train[train_data_matrix.nonzero()]))

bybrid_train_label = train_data_matrix[train_data_matrix.nonzero()]

hybrid_test_feature = np.column_stack((memory_based_cf_user_all[test_data_matrix.nonzero()], \
										memory_based_cf_item_all[test_data_matrix.nonzero()], \
										sgd_all[test_data_matrix.nonzero()], \
										svd_all[test_data_matrix.nonzero()], \
										tbr_preference_test[test_data_matrix.nonzero()], 
										tbr_similarity_test[test_data_matrix.nonzero()]))

bybrid_test_label = test_data_matrix[test_data_matrix.nonzero()]

print 'Perform the linear regression...'

regr = linear_model.LinearRegression()
regr.fit(hybrid_train_feature, bybrid_train_label)

train_prediction = regr.predict(hybrid_train_feature)
test_prediction = regr.predict(hybrid_test_feature)

print 'training rmse:', utils.list_rmse(train_prediction, bybrid_train_label)
print 'testing rmse:', utils.list_rmse(test_prediction, bybrid_test_label)