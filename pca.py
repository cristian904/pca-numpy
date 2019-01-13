import numpy as np
from numpy import linalg

A = np.array([[1, 2], [3, 4], [5, 6]])

#calculate the mean of each column
M = np.mean(A.T, axis=1)
print(M)

#center the columns by subtracting column means
C = A - M
print(C)

#calculate covariance matrix of centered matrix
V = np.cov(C.T)
print(V)

#eigendecomposition of covariance matrix
values, vectors = linalg.eig(V)
print(vectors)
print(values)

#project data
P = vectors.T.dot(C.T)
print(P.T)

###################From library#################

from sklearn.decomposition import PCA

pca =PCA(2)
pca.fit(A)
print(pca.components_)
print(pca.explained_variance_)
#transform data
B = pca.transform(A)
print(B)