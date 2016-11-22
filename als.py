import scipy.sparse as sp
import numpy as np
from scipy.sparse.linalg import svds
from math import sqrt
from sklearn.metrics import mean_squared_error
import load_matrix
import matplotlib.pyplot as plt

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

def rmse(I,R,Q,P):
    return np.sqrt(np.sum((I * (R - prediction(P,Q)))**2)/len(I[I > 0]))

lmbda = 0.1 # Regularisation weight
k = 20 # Dimensionality of latent feature space
m, n = R.shape # Number of users and items
n_epochs = 15 # Number of epochs

P = 3 * np.random.rand(k,m) # Latent user feature matrix
Q = 3 * np.random.rand(k,n) # Latent movie feature matrix
Q[0,:] = R[R != 0].mean(axis=0) # Avg. rating for each movie
E = np.eye(k) # (k x k)-dimensional idendity matrix

train_errors = []
test_errors = []

# Repeat until convergence
for epoch in range(n_epochs):
	print epoch
    # Fix Q and estimate P
	for i, Ii in enumerate(I):
		nui = np.count_nonzero(Ii) # Number of items user i has rated
		if (nui == 0): nui = 1 # Be aware of zero counts!

		# Least squares solution
		Ai = np.dot(Q, np.dot(np.diag(Ii), Q.T)) + lmbda * nui * E
		Vi = np.dot(Q, np.dot(np.diag(Ii), R[i].T))
		P[:,i] = np.linalg.solve(Ai,Vi)
        
    # Fix P and estimate Q
	for j, Ij in enumerate(I.T):
		nmj = np.count_nonzero(Ij) # Number of users that rated item j
		if (nmj == 0): nmj = 1 # Be aware of zero counts!

		# Least squares solution
		Aj = np.dot(P, np.dot(np.diag(Ij), P.T)) + lmbda * nmj * E
		Vj = np.dot(P, np.dot(np.diag(Ij), R[:,j]))
		Q[:,j] = np.linalg.solve(Aj,Vj)
    
	train_rmse = rmse(I,R,Q,P)
	test_rmse = rmse(I2,T,Q,P)
	train_errors.append(train_rmse)
	test_errors.append(test_rmse)
    
	print "[Epoch %d/%d] train error: %f, test error: %f" \
	%(epoch+1, n_epochs, train_rmse, test_rmse)
    
print "Algorithm converged"


# Check performance by plotting train and test errors
import matplotlib.pyplot as plt

plt.plot(range(n_epochs), train_errors, label='Training Data');
plt.plot(range(n_epochs), test_errors, label='Testing Data');
plt.title('Collaborative Filtering by Alternating Least Squares Learning Curve')
plt.xlabel('Iterations');
plt.ylabel('RMSE');
plt.legend()
plt.grid()
plt.show()