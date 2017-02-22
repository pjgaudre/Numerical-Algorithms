import numpy as np
from copy import copy, deepcopy
def forward(A,b):
    try:
        n,m = A.shape
    except:
        print "Matrix must be two dimensional array."
        return
    if n != m:
        print "Matrix must be square"
        return
    elif n != len(b):
        print "Array must have same number of rows as column vector"
        return
    elif np.any(np.triu(A,k=1)) or np.any(np.diag(A)==0.0):
        print "Array must be lower triangular with nonzero diagonal elements"
        return
    x = copy(b)
    for i in xrange(n):
        for j in xrange(i):
            x[i] -= A[i,j]*x[j]
        x[i] /= A[i,i]
    return x

def forward_test():
    np.random.seed(3)
    n=10
    A = np.random.randn(n,n)
    A = A - np.triu(A,k=1) + n*np.diag(np.random.randn(n))
    b = np.random.rand(n)
    xstar = forward(A,b)
    Error = np.linalg.norm(A.dot(xstar)-b)/np.linalg.norm(b)
    if Error <=1e-10:
        print "Test passed"
    else:
        print "test failed"


def back(A,b):
    try:
        n,m = A.shape
    except:
        print "Matrix must be two dimensional array."
        return
    if n != m:
        print "Matrix must be square"
        return
    elif n != len(b):
        print "Array must have same number of rows as column vector"
        return
    elif np.any(np.tril(A,k=-1)) or np.any(np.diag(A)==0.0):
        print "Array must be upper triangular with nonzero diagonal elements"
        return
    x = copy(b)
    for i in reversed(xrange(n)):
        for j in xrange(i+1,n):
            x[i] -= A[i,j]*x[j]
        x[i] /= A[i,i]
    return x

def back_test():
    np.random.seed(3)
    n=10
    A = np.random.randn(n,n)
    A = A - np.tril(A,k=-1) + n*np.diag(np.random.randn(n))
    b = np.random.rand(n)
    xstar = back(A,b)
    Error = np.linalg.norm(A.dot(xstar)-b)/np.linalg.norm(b)
    if Error <=1e-10:
        print "Test passed"
    else:
        print "test failed"

back_test()
