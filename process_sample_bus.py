import json
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import GradientBoostingClassifier
def load_data():

    data = []
    label = []

    with open('data/yelp_academic_dataset_business.json') as data_file:
        for line in data_file:
            raw = json.loads(line)
            label.append(raw['stars'])
            this_data = []

            this_data.append(raw['review_count'])
            attrs = raw['attributes']

            this_data.append(1 if attrs.get('Take-out', False) else 0)
            this_data.append(1 if attrs.get('Wi-Fi', '') == 'free' else 0)
            this_data.append(attrs.get('Price Range', 0))
            this_data.append(1 if attrs.get('Accepts Credit Cards', False) else 0)
            this_data.append(1 if attrs.get('Good for Kids', False) else 0)
            this_data.append(1 if attrs.get('Good for Groups', False) else 0)
            this_data.append(1 if attrs.get('Alcohol', 'none') == 'none' else 0)
            this_data.append(1 if attrs.get('Outdoor Seating', False) else 0)
            this_data.append(1 if True in attrs.get('Parking', {}).keys() else 0)
            this_data.append(1 if attrs.get('Has TV', False) else 0)
            this_data.append(1 if attrs.get('Outdoor Seating', False) else 0)
            this_data.append(1 if attrs.get('Waiter Service', False) else 0)

            data.append(this_data)


    data = np.array(data)
    label = np.array(label)

    mean_label = np.mean(label)
    label = np.array([1 if score > mean_label else 0 for score in label])

    #return (data[:800], label[:800], data[800:], label[800:])

    train_size = len(label) * 8 / 10

    def model(clf, name):
        clf.fit(data[:train_size], label[:train_size])
        print name, 'test accu', (sum(clf.predict(data[train_size:]) == label[train_size:]) * 1.0 / (len(label) - train_size))
        print name, 'train accu', (sum(clf.predict(data[:train_size]) == label[:train_size]) * 1.0 / train_size)

    model(GaussianNB(), 'nb')
    model(LogisticRegression(), 'logis')
    model(GradientBoostingClassifier(), 'bo')
