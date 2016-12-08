import json
import utils
import load_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.kernel_ridge import KernelRidge
import matplotlib.pyplot as plt
import math
import numpy as np

file_city_name = utils.file_city_name
overall_matrix = load_matrix.test_data_matrix + load_matrix.train_data_matrix

# '_cosine_similarities.np' or '_user_preference_cos.np'
sorting_method = '_cosine_similarities.np'

cosine_similarities = np.load('data/' + file_city_name + sorting_method)

# test_predictions = np.zeros((load_matrix.n_users, load_matrix.n_items))
# train_predictions = np.zeros((load_matrix.n_users, load_matrix.n_items))

overall_predictions = np.zeros((load_matrix.n_users, load_matrix.n_items))


# def make_predictions(testing_nonzero_index, reverse_index, destination_matrix, clf):

# 	# for each of the non-zero real restaurant from the test set...
# 	testing_sorted_index = [reverse_index[old_index] for old_index in testing_nonzero_index]
# 	testing_sorted_index_ = np.array(testing_sorted_index).reshape(len(testing_sorted_index), 1)
# 	predictions = clf.predict(testing_sorted_index_)
# 	predictions = np.maximum(1, np.minimum(predictions, 5))

# 	destination_matrix[user_idx, testing_nonzero_index] = predictions.reshape(1, len(testing_sorted_index))	

for user_idx in range(load_matrix.n_users):
	# print 'making predictions for the user:', user_idx

	# list of (similarity, real restaurant index), sorted by similarity
	list_with_index = [(sim, rest_index) for (rest_index, sim) in enumerate(cosine_similarities[user_idx,])]
	list_with_index.sort()

	# list of (real restaurant index, the index of that restaurant in the sorted similarity list),
	# this list is sorted by the real restaurant index
	# use this to convert form the old real restaurant index, into the new index in the list_with_index 
	reverse_index = {old_index: new_index for (new_index, (_, old_index)) in enumerate(list_with_index)}
	# reverse_index.sort()

	# the real index of non-zero-rating restaurant
	nonzero_index = np.nonzero(load_matrix.train_data_matrix[user_idx,])[0]

	# the ratings of this user, listed in the original order in the training matrix
	nonzero_values = load_matrix.train_data_matrix[user_idx, nonzero_index]

	# for each of the non-zero real restaurant index, the corresponding new index 
	sorted_index = [reverse_index[old_index] for old_index in nonzero_index]
	
	sorted_index = np.array(sorted_index).reshape(len(sorted_index), 1)
	nonzero_values = np.array(nonzero_values).reshape(len(nonzero_values), 1)

	clf = KernelRidge(alpha=1.0)
	clf.fit(sorted_index, nonzero_values)

	# make predictions
	overall_nonzero_index = np.nonzero(overall_matrix[user_idx,])[0]
	print len(overall_nonzero_index)
	new_index = [reverse_index[old_index] for old_index in overall_nonzero_index]
	new_index = np.array(new_index).reshape(len(new_index), 1)
	pred = clf.predict(new_index)
	pred = np.maximum(1, np.minimum(pred, 5))
	print pred
	overall_predictions[user_idx, overall_nonzero_index] = pred.reshape(1, len(overall_nonzero_index))	

	# if len(np.nonzero(load_matrix.test_data_matrix[user_idx,])[0]) > 0:
	# 	testing_nonzero_index = np.nonzero(load_matrix.test_data_matrix[user_idx,])[0]
	# 	make_predictions(testing_nonzero_index, reverse_index, test_predictions, clf)

	# make_predictions(nonzero_index, reverse_index, train_predictions, clf)

print 'testing rmse:', utils.rmse(overall_predictions, load_matrix.test_data_matrix)
print 'training rmse:', utils.rmse(overall_predictions, load_matrix.train_data_matrix)

# save the predictions
# with open('data/' + file_city_name + sorting_method + '_train_predictions.np', 'w') as file:
# 	np.save(file, train_predictions)

# with open('data/' + file_city_name + sorting_method + '_test_predictions.np', 'w') as file:
# 	np.save(file, test_predictions)


with open('data/' + file_city_name + sorting_method + '_all_predictions.np', 'w') as file:
	np.save(file, overall_predictions)


