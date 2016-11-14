import json

file_city_name = 'las_vegas'
city_name = 'Las Vegas'

businesses = set()

print 'processing bussiness, find the restaurants in the city'

with open('data/yelp_academic_dataset_business.json') as data_file:
    for line in data_file:
        buss = json.loads(line)
        if buss['city'] == city_name:
            businesses.add(buss['business_id'])


# we have a file that contains all of the information of reviews of las vegas
print 'extracting all the reviews from the city'
all_output = open('data/' + file_city_name + '_reviews.json', 'w')

with open('data/yelp_academic_dataset_review.json') as data_file:
    for line in data_file:
        review = json.loads(line)
        if review['business_id'] in businesses:
            all_output.write(line)

all_output.close()



# for now maybe we need a skinny version of the data, with each line
# only contains the following informaiton:
# user_id, bussiness_id, rating
print 'constructing skinny files'
skinny_output = open('data/' + file_city_name + '_reviews_ratings_only.txt', 'w')

with open('data/' + file_city_name + '_reviews.json') as data_file:
    for line in data_file:
        review = json.loads(line)
        skinny_output.write(review['user_id'] + '\t' + review['business_id'] + '\t' + str(review['stars']) + '\n')

skinny_output.close()

