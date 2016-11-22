# CS229-project-yelp

Some links that might be useful...

## Collaborative Filtering:

[Matrix Factorization: A Simple Tutorial](http://www.quuxlabs.com/blog/2010/09/matrix-factorization-a-simple-tutorial-and-implementation-in-python/#basic-ideas)

[Tutorial on Collaborative Filtering and Matrix Factorization](https://lazyprogrammer.me/tutorial-on-collaborative-filtering-and-matrix-factorization-in-python/)

[Intro to Recommender Systems: Collaborative Filtering](http://blog.ethanrosenthal.com/2015/11/02/intro-to-collaborative-filtering/)

[Implementing your own recommender systems in Python](http://online.cambridgecoding.com/notebooks/eWReNYcAfB/implementing-your-own-recommender-systems-in-python-2)

[->Stochastic gradient descent](http://online.cambridgecoding.com/notebooks/mhaller/implementing-your-own-recommender-systems-in-python-using-stochastic-gradient-descent-4#implementing-your-own-recommender-systems-in-python-using-stochastic-gradient-descent)

[->Alternating Least Squares](http://online.cambridgecoding.com/notebooks/mhaller/predicting-user-preferences-in-python-using-alternating-least-squares)

## Content based recommendation system:

[Beginners Guide to learn about Content Based Recommender Engines](https://www.analyticsvidhya.com/blog/2015/08/beginners-guide-learn-content-based-recommender-systems/)

## Code Usage

1. `utils.py` defines the city we're currently dealing with. 
..* `file_city_name` defines the city name used to all of the intermediate files
..* `city_name` has to match the city name stored within the Yelp dataset

2. `extract_las_vegas_ratings.py` 
..* Find all the reviews from the city
..* Filter the users and restaurants with too few reviews
..* Construct the skinny file which only include the `user_id`, `business_id` and the rating

3. `extract_matrix_index.py` defines the mappings from the `user_id` and `business_id` in the dataset, into integers from 0, used as the index for the data matrix

4. `load_matrix.py` should be imported for all of the learning algorithms, to load the training and testing data matrix, with the consistent index

5. `concat_reviews.py` concatenate all the reviews for an user or a restaurant into a long sentence

6. `dump_bag_of_words.py` use the concatenated reviews and construct the tf-idf vectors for each user and restaurant, should be called once before running any text-based or review-based algorithms

7. `dump_user_preference.py` constructs the user preference matrix by computing the linear combination of restaurant feature vectors for each user, using the user's ratings as the weights, should be called at least once before running the review-based algorithm
