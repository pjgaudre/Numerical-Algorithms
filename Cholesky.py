import numpy as np
from copy import copy
import warnings
warnings.filterwarnings('error')

def Cholesky(A):
    '''
    Cholesky(A) produces a lower triangular matrix L satisfying the equation L*L.T=A.
    The chol function assumes that A is symmetric and positive definite.
    '''
    try:
        n,m = A.shape
    except:
        print "Array must be a two dimensioal."
        return
    if n!=m:
        print "Matrix must be square."
        return
    if not (A.transpose() == A).all():
        print "Matrix must be symmetric."
        return
    try:
        for k in xrange(n):
            A[k,k] = np.sqrt(A[k,k])
            for i in xrange(k+1,n):
                A[i,k] /= A[k,k]
            for j in xrange(k+1,n):
                for i in xrange(j,n):
                    A[i,j] -= A[i,k]*A[j,k]
        return np.tril(A)
    except:
        print "LinAlgError: Matrix is not positive definite - Cholesky decomposition cannot be computed"
        return
