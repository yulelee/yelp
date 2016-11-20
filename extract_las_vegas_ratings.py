import json
import utils

file_city_name = utils.file_city_name
city_name = utils.city_name

businesses = set()

print 'processing business, find the restaurants in the city'

with open('data/yelp_academic_dataset_business.json') as data_file:
    for line in data_file:
        buss = json.loads(line)
        if buss['city'] == city_name:
            businesses.add(buss['business_id'])


# we have a file that contains all of the information of reviews of [las vegas]
# this new file would contain all of the reviews from this city
print 'extracting all the reviews from the city'
all_output = open('data/' + file_city_name + '_reviews.json', 'w')

user_count = {}
restaurant_count = {}

with open('data/yelp_academic_dataset_review.json') as data_file:
    for line in data_file:
        review = json.loads(line)
        if review['business_id'] in businesses:
            all_output.write(line)

            # record how many time a user has appeared
            user_id, restaurant_id = review['user_id'], review['business_id']
            if user_id not in user_count: user_count[user_id] = 0
            user_count[user_id] += 1
            if restaurant_id not in restaurant_count: restaurant_count[restaurant_id] = 0
            restaurant_count[restaurant_id] += 1

all_output.close()

# filter the users based on the number of reviews he/she has written
NUM_THRESH = util.NUM_REVIEWS_THRESH
eligible_users = [user_id for (user_id, count) in user_count.items() if count >= NUM_THRESH]
eligible_restaurants = [rest_id for (rest_id, count) in restaurant_count.items() if count >= NUM_THRESH]
print len(eligible_users), 'users are eligible, out of a total of', len(user_count)
print len(eligible_restaurants), 'restaurants are eligible, out of a total of', len(restaurant_count)
eligible_users = set(eligible_users)
eligible_restaurants = set(eligible_restaurants)
filter_output = open('data/' + file_city_name + '_reviews_filtered.json', 'w')
total_count, eligible_review_count = 0, 0
with open('data/' + file_city_name + '_reviews.json') as data_file:
    for line in data_file:
        total_count += 1
        review = json.loads(line)
        if review['user_id'] in eligible_users and review['business_id'] in eligible_restaurants:
            filter_output.write(line)
            eligible_review_count += 1

print eligible_review_count, 'reviews eligible, out of a total of', total_count

filter_output.close()

# for now maybe we need a skinny version of the data, with each line
# only contains the following informaiton:
# user_id, business_id, rating
print 'constructing skinny files'
skinny_output = open('data/' + file_city_name + '_reviews_ratings_only.txt', 'w')
with open('data/' + file_city_name + '_reviews_filtered.json') as data_file:
    for line in data_file:
        review = json.loads(line)
        skinny_output.write(review['user_id'] + '\t' + review['business_id'] + '\t' + str(review['stars']) + '\n')

skinny_output.close()

