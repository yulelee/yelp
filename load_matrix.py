import numpy as np
from sklearn import cross_validation as cv
import pandas as pd
import utils
import json

file_city_name = utils.file_city_name

header = ['user_id', 'business_id', 'rating']
df = pd.read_csv('data/' + file_city_name + '_reviews_ratings_only.txt', sep='\t', names=header)

n_users = df.user_id.unique().shape[0]
n_items = df.business_id.unique().shape[0]

print 'Number of users = ' + str(n_users) + ' | Number of items = ' + str(n_items)

print 'load the index mappings'
with open('data/' + file_city_name + '_filtered_user_index.json') as user_idx_file:
    user_id_map = json.load(user_idx_file)
with open('data/' + file_city_name + '_filtered_restaurants_index.json') as rest_idx_file:
    item_id_map = json.load(rest_idx_file)

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


