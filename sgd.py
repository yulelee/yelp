import scipy.sparse as sp
import numpy as np
from scipy.sparse.linalg import svds
from math import sqrt
from sklearn.metrics import mean_squared_error
import load_matrix
import matplotlib.pyplot as plt
import utils

file_city_name = utils.file_city_name

R = load_matrix.train_data_matrix
T = load_matrix.test_data_matrix

# Index matrix for training data
I = R.copy()
I[I > 0] = 1
I[I == 0] = 0

# Index matrix for test data
I2 = T.copy()
I2[I2 > 0] = 1
I2[I2 == 0] = 0

# Predict the unknown ratings through the dot product of the latent features for users and items 
def prediction(P,Q):
    return np.dot(P.T,Q)

lmbda = 0.1 # Regularisation weight
k = 50  # Dimension of the latent feature space
m, n = R.shape  # Number of users and items
n_epochs = 50  # Number of epochs
gamma = 0.01  # Learning rate

P = 1 * np.random.rand(k,m) # Latent user feature matrix
Q = 1 * np.random.rand(k,n) # Latent movie feature matrix

# Calculate the RMSE
def rmse(I,R,Q,P):
    return np.sqrt(np.sum((I * (R - prediction(P,Q)))**2)/len(I[I > 0]))

train_errors = []
test_errors = []

#Only consider non-zero matrix 
users,items = R.nonzero()      
for epoch in xrange(n_epochs):
    for u, i in zip(users,items):
        e = R[u, i] - prediction(P[:,u],Q[:,i])  # Calculate error for gradient
        P[:,u] += gamma * ( e * Q[:,i] - lmbda * P[:,u]) # Update latent user feature matrix
        Q[:,i] += gamma * ( e * P[:,u] - lmbda * Q[:,i])  # Update latent movie feature matrix
    # train_rmse = rmse(I,R,Q,P) # Calculate root mean squared error from train dataset
    # test_rmse = rmse(I2,T,Q,P) # Calculate root mean squared error from test dataset
    # train_errors.append(train_rmse)
    # test_errors.append(test_rmse)
    print epoch
    # print train_rmse
    # print test_rmse

final_result = prediction(P, Q)
print 'test_error:', utils.rmse(final_result, T)

with open('data/' + file_city_name + 'sgd' + '_all_predictions.np', 'w') as file:
    np.save(file, final_result)


# Check performance by plotting train and test errors

# plt.plot(range(n_epochs), train_errors, label='Training');
# plt.plot(range(n_epochs), test_errors, label='Testing');
# plt.title('Collaborative Filtering by Gradient Descent Learning Curve')
# plt.xlabel('Iterations');
# plt.ylabel('RMSE');
# plt.legend()
# plt.grid()
# plt.show()
