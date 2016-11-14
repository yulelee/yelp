# This file extract the city information of all of the businesses,
# because the restaurants are from different cities, processing them
# all together seems not to be a good idea, here we can find
# the count of all the cities, and pick the cities with the most restaurants.

import json

cities = {}

with open('data/yelp_academic_dataset_business.json') as data_file:
    for line in data_file:
        buss = json.loads(line)
        city = buss['city']
        if city not in cities: cities[city] = 0
        cities[city] += 1

city_list = [(count, name) for (name, count) in cities.items()]
city_list.sort()
for city in city_list:
    print city
