# Run linear regression on all of the results

import utils
import load_matrix
import math
import numpy as np
from sklearn import linear_model
from sklearn import kernel_ridge
from sklearn import svm

from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt

file_city_name = utils.file_city_name

train_data_matrix = load_matrix.train_data_matrix
test_data_matrix = load_matrix.test_data_matrix

plot_curve = False
plot_step = True

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
tbr_preference_all = np.load('data/' + file_city_name + sorting_method + '_all_predictions.np')
# tbr_preference_test = np.load('data/' + file_city_name + sorting_method + '_test_predictions.np')

sorting_method = '_cosine_similarities.np'
tbr_similarity_all = np.load('data/' + file_city_name + sorting_method + '_all_predictions.np')
# tbr_similarity_test = np.load('data/' + file_city_name + sorting_method + '_test_predictions.np')


print 'Constructing the features and labels...'

# construct the features and labels for the linear regression
hybrid_train_feature = np.column_stack((memory_based_cf_user_all[train_data_matrix.nonzero()], \
										memory_based_cf_item_all[train_data_matrix.nonzero()], \
										np.maximum(1, np.minimum(5, sgd_all[train_data_matrix.nonzero()])), \
										# svd_all[train_data_matrix.nonzero()], \
										tbr_preference_all[train_data_matrix.nonzero()], 
										tbr_similarity_all[train_data_matrix.nonzero()]))

hybrid_train_label = train_data_matrix[train_data_matrix.nonzero()]

if plot_curve:
	plt.plot(range(len(hybrid_train_label)), np.sort(hybrid_train_label), label='Labels');
	plt.plot(range(len(hybrid_train_label)), np.sort(hybrid_train_feature[:,0]), label='Feature1');
	plt.plot(range(len(hybrid_train_label)), np.sort(hybrid_train_feature[:,1]), label='Feature2');
	plt.plot(range(len(hybrid_train_label)), np.sort(hybrid_train_feature[:,2]), label='Feature3');
	plt.plot(range(len(hybrid_train_label)), np.sort(hybrid_train_feature[:,3]), label='Feature4');
	plt.plot(range(len(hybrid_train_label)), np.sort(hybrid_train_feature[:,4]), label='Feature5');


hybrid_test_feature = np.column_stack((memory_based_cf_user_all[test_data_matrix.nonzero()], \
										memory_based_cf_item_all[test_data_matrix.nonzero()], \
										np.maximum(1, np.minimum(5, sgd_all[test_data_matrix.nonzero()])), \
										# svd_all[test_data_matrix.nonzero()], \
										tbr_preference_all[test_data_matrix.nonzero()], 
										tbr_similarity_all[test_data_matrix.nonzero()]))

hybrid_test_label = test_data_matrix[test_data_matrix.nonzero()]

print 'Perform the linear regression...'

print 'Linear Regression'

print hybrid_train_feature.shape
poly = PolynomialFeatures(3)
hybrid_train_feature = poly.fit_transform(hybrid_train_feature)
hybrid_test_feature = poly.fit_transform(hybrid_test_feature)

print hybrid_train_feature.shape

# regr = linear_model.LinearRegression()
regr = linear_model.LinearRegression()
regr.fit(hybrid_train_feature, hybrid_train_label)

train_prediction = regr.predict(hybrid_train_feature)
test_prediction = regr.predict(hybrid_test_feature)

if plot_curve:
	plt.plot(range(len(hybrid_train_label)), np.sort(train_prediction), label='predictions');
	plt.legend()
	plt.grid()
	plt.show()

def plot_step_helper(preds, labels):
	label_with_index = [(label, index) for (index, label) in enumerate(labels)]
	label_with_index.sort()
	reverse_index = [old_index for (_, old_index) in label_with_index]

	one = np.sort(preds[np.where(labels == 1)])
	two = np.sort(preds[np.where(labels == 2)])
	three = np.sort(preds[np.where(labels == 3)])
	four = np.sort(preds[np.where(labels == 4)])
	five = np.sort(preds[np.where(labels == 5)])
	sorted_predctions = np.concatenate((one, two, three, four, five))


	plt.plot(range(len(labels)), sorted_predctions, label='predictions');
	plt.plot(range(len(labels)), labels[reverse_index], label='label')
	plt.legend()
	plt.grid()
	plt.show()

if plot_step:
	plot_step_helper(train_prediction, hybrid_train_label)
	plot_step_helper(test_prediction, hybrid_test_label)

print 'training rmse:', utils.list_rmse(train_prediction, hybrid_train_label)
print 'testing rmse:', utils.list_rmse(test_prediction, hybrid_test_label)

# print 'SVM'

# regr = kernel_ridge.KernelRidge(kernel = 'rbf')
# regr.fit(hybrid_train_feature, bybrid_train_label)

# train_prediction = regr.predict(hybrid_train_feature)
# test_prediction = regr.predict(hybrid_test_feature)

# print 'training rmse:', utils.list_rmse(train_prediction, bybrid_train_label)
# print 'testing rmse:', utils.list_rmse(test_prediction, bybrid_test_label)