# save the vectorized reviews to future use

import json
import utils
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import numpy as np

file_city_name = utils.file_city_name

# get the corpus
with open('data/' + file_city_name + '_users_concat_reviews.txt') as data_file:
	user_corpus = json.load(data_file)

with open('data/' + file_city_name + '_restaurants_concat_reviews.txt') as data_file:
	restaurant_corpus = json.load(data_file)

# directly to tf-idf matrix

vectorizer = TfidfVectorizer(min_df = 1, stop_words = 'english')

print 'constructing tf-idf matrix for the restaurants'
restaurants_X = vectorizer.fit_transform(restaurant_corpus)

print 'constructing tf-idf matrix for the users'
user_X = vectorizer.transform(user_corpus)

print 'computing similarities'
cosine_similarities = linear_kernel(user_X, restaurants_X)

with open('data/' + file_city_name + '_user_X.np', 'w') as file:
	np.save(file, user_X)

with open('data/' + file_city_name + '_restaurants_X.np', 'w') as file:
	np.save(file, restaurants_X)

with open('data/' + file_city_name + '_cosine_similarities.np', 'w') as file:
	np.save(file, cosine_similarities)