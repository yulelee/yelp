# use the vectorized reviews for the restaurants, extract the 'preference' vector 
# of each users

import json
import utils
import load_matrix
import numpy as np
from sklearn.metrics.pairwise import linear_kernel
from sklearn.preprocessing import normalize

file_city_name = utils.file_city_name

restaurants_X = np.load('data/' + file_city_name + '_restaurants_X.np')
restaurants_X = restaurants_X.item()
train_matrix = load_matrix.train_data_matrix

print 'computing the dot product'
user_preference = np.zeros((train_matrix.shape[0], restaurants_X.shape[1]))
for i in range(train_matrix.shape[0]):
	nonzero_index = np.nonzero(train_matrix[i,])[0]
	nonzero_values = train_matrix[i, nonzero_index]
	user_preference[i,] = nonzero_values.dot(restaurants_X[nonzero_index,].toarray())

with open('data/' + file_city_name + '_user_preference.np', 'w') as file:
	np.save(file, user_preference)

# normalize user_preference
print 'normalize'
user_preference = normalize(user_preference)

print 'computing the similarity'
cosine_similarities = linear_kernel(user_preference, restaurants_X)

with open('data/' + file_city_name + '_user_preference_cos.np', 'w') as file:
	np.save(file, cosine_similarities)

