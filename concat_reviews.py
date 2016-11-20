import utils
import json

file_city_name = utils.file_city_name

# load the index mappings
with open('data/' + file_city_name + '_filtered_user_index.json') as user_idx_file:
    user_id_map = json.load(user_idx_file)
with open('data/' + file_city_name + '_filtered_restaurants_index.json') as rest_idx_file:
    item_id_map = json.load(rest_idx_file)

user_concat = ['' for _ in range(len(user_id_map))]
restaurant_concat = ['' for _ in range(len(item_id_map))]

# concat all of the reviews, all of the reviews of the same user are concatenated together,
# same with the restaurants
with open('data/' + file_city_name + '_reviews_filtered.json') as data_file:
	for line in data_file:
	    review = json.loads(line)
	    text, user_id, restaurant_id = review['text'], review['user_id'], review['business_id']
	    user_concat[user_id_map[user_id]] += ' ' + text
	    restaurant_concat[item_id_map[restaurant_id]] += ' ' + text

# those 2 files contains the list of concatenated reviews, the index is the same
# as what is in the index mappings 
with open('data/' + file_city_name + '_users_concat_reviews.txt', 'w') as data_file:
	json.dump(user_concat, data_file)

with open('data/' + file_city_name + '_restaurants_concat_reviews.txt', 'w') as data_file:
	json.dump(restaurant_concat, data_file)