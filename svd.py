import scipy.sparse as sp
import numpy as np
from scipy.sparse.linalg import svds
from math import sqrt
from sklearn.metrics import mean_squared_error
import load_matrix
import utils

file_city_name = utils.file_city_name

train_data_matrix = load_matrix.train_data_matrix
test_data_matrix = load_matrix.test_data_matrix

#get SVD components from train matrix. Choose k.
u, s, vt = svds(train_data_matrix, k = 200)
s_diag_matrix = np.diag(s)
X_pred = np.dot(np.dot(u, s_diag_matrix), vt)

print 'Training User-based CF RMSE: ' + str(utils.rmse(X_pred, train_data_matrix))
print 'Testing User-based CF RMSE: ' + str(utils.rmse(X_pred, test_data_matrix))

with open('data/' + file_city_name + 'svd' + '_all_predictions.np', 'w') as file:
    np.save(file, X_pred)
