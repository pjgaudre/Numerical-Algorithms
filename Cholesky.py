import numpy as np

def Cholesky(A):
    try:
        n,m = A.shape
        if n!=m:
            print "Matrix must be square"
            return
    except:
        print "Array must be a two dimensioal"
    for k in xrange(n):
        A[k,k] = np.sqrt(A[k,k])
        for i in xrange(k+1,n):
            A[i,k] /= A[k,k]
        for j in xrange(k+1,n):
            for i in xrange(j,n):
                A[i,j] -= A[i,k]*A[j,k]
    return np.triu(A)
np.random.seed(10)
A = np.random.rand(n,n)
A = (A + A.T)/2.0
A = A.dot(A.T)+np.diag([1.0*n for i in range(len(b))])
L = Cholesky(A)
np.random.seed(10)
A = np.random.rand(n,n)
A = (A + A.T)/2.0
A = A.dot(A.T)+np.diag([1.0*n for i in range(len(b))])
L.T.dot(L)-A
