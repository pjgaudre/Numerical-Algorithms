import numpy as np
from copy import copy

def LU(A):
    '''
    This algorithm is the LU decomposition of a matrix WITHOUT PIVOTING.
    Input:    A  :   Matrix of size n by n.
    Ouput:    L  :   Lowertriangular matrix of size n by n.  (L*U = A)
              U  :   Uppertriangular matrix of size n by n.
    '''
    try:
         n,m = A.shape
    except:
        print "Matrix must be two dimensional."
        return
    if n!=m:
        print "Matrix must be square."
        return
    U = copy(A)
    L = np.eye(n)
    for j in xrange(n-1):
        for k in xrange(j+1,n):
            L[k,j] = U[k,j]/U[j,j]
            U[k,j:] = U[k,j:]-L[k,j]*U[j,j:]
    return L,U

def LUP(A):
    '''
    This algorithm is the LU decomposition of a matrix WITH PIVOTING.
    Input:    A  :   Matrix of size n by n.
    Ouput:    L  :   Lowertriangular matrix of size n by n.
              U  :   Uppertriangular matrix of size n by n.
              P  :   Permutation vector
              where  L*U = A[P,:]
    '''
    try:
         n,m = A.shape
    except:
        print "Matrix must be two dimensional."
        return
    if n!=m:
        print "Matrix must be square."
        return
    P  = np.arange(n)
    L  = np.eye(n)
    U  = copy(A)
    for j in xrange(n-1):
        i = np.argmax(abs(U[j:,j])) + j
        U[[j, i],j:] = U[[i, j],j:]
        L[[i, j],:j] = L[[j, i],:j];
        P[[i, j]] = P[[j, i]]
        for k in xrange(j+1,n):
            L[k,j] = U[k,j]/U[j,j]
            U[k,j:] = U[k,j:] - L[k,j]*U[j,j:]
    return L, U, P


def GE_test():
    n = 10
    A = np.random.randn(n,n)
    A = (A.T + A)/2.0
    L,U = GE(A)
    if np.linalg.norm(L.dot(U)-A,'fro') <1.e-14:
        print "test passed"
    else:
        print "test failed"
GE_test()

def GEP_test():
    n = 10
    A = np.random.randn(n,n)
    L,U,P = GEP(A)
    if np.linalg.norm(L.dot(U)-A[P,:],'fro') <1.e-14:
        print "test passed"
    else:
        print "test failed"
GEP_test()
