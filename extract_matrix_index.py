import utils
import json

file_city_name = utils.file_city_name

# constructing the mappings from user_id and restaurant_id into matrix indices
unique_user_counter, unique_rest_counter = 0, 0
user_index_map, rest_index_map = {}, {}

with open('data/' + file_city_name + '_reviews_filtered.json') as data_file:
    for line in data_file:
        review = json.loads(line)
        if review['user_id'] not in user_index_map.keys(): 
            user_index_map[review['user_id']] = unique_user_counter
            unique_user_counter += 1
        if review['business_id'] not in rest_index_map.keys():
            rest_index_map[review['business_id']] = unique_rest_counter
            unique_rest_counter += 1

print 'unique users:', unique_user_counter
print 'unique restaurants:', unique_rest_counter

# save the index mappings 
with open('data/' + file_city_name + '_filtered_user_index.json', 'w') as user_idx_file:
    json.dump(user_index_map, user_idx_file)
with open('data/' + file_city_name + '_filtered_restaurants_index.json', 'w') as rest_idx_file:
    json.dump(rest_index_map, rest_idx_file)