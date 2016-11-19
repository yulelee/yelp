import numpy as np
from sklearn import cross_validation as cv
import pandas as pd

file_city_name = 'las_vegas'

header = ['user_id', 'business_id', 'rating']
df = pd.read_csv('data/' + file_city_name + '_reviews_ratings_only.txt', sep='\t', names=header)

n_users = df.user_id.unique().shape[0]
n_items = df.business_id.unique().shape[0]

print 'Number of users = ' + str(n_users) + ' | Number of items = ' + str(n_items)

print 'create mappings for the user_id and item_id into matrix index'
user_count, item_count = 0, 0
user_id_map, item_id_map = {}, {}
for user_id in df.user_id.unique():
    user_id_map[user_id] = user_count
    user_count += 1
for item_id in df.business_id.unique():
    item_id_map[item_id] = item_count
    item_count += 1

print 'split and construct matrix'
train_data, test_data = cv.train_test_split(df, test_size=0.25)

train_data_matrix = np.zeros((n_users, n_items))

total_count = 0

for line in train_data.itertuples():
	train_data_matrix[user_id_map[line[1]], item_id_map[line[2]]] = line[3]  
	total_count += 1

test_data_matrix = np.zeros((n_users, n_items))
for line in test_data.itertuples():
    test_data_matrix[user_id_map[line[1]], item_id_map[line[2]]] = line[3]
    total_count += 1

print total_count / (n_users * n_items * 1.0)


