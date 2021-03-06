import numpy as np
from copy import copy, deepcopy
def forward(L,b):
    '''
    Solves the system Lx=b by forward elimination. L is a lower triangular matrix
    with with nonzero diagonal elements
    INPUT: L , array lower triangular matrix with nonzero diagonal elements.
           b, RHS vector
    OUTPUT : x, vector , solution to Lx=b
    '''
    try:
        n,m = L.shape
    except:
        print "Matrix must be two dimensional array."
        return
    if n != m:
        print "Matrix must be square"
        return
    elif n != len(b):
        print "Array must have same number of rows as column vector"
        return
    elif np.any(np.triu(L,k=1)) or np.any(np.diag(L)==0.0):
        print "Array must be lower triangular with nonzero diagonal elements"
        return
    x = copy(b)
    for i in xrange(n):
        for j in xrange(i):
            x[i] -= L[i,j]*x[j]
        x[i] /= L[i,i]
    return x



def back(U,b):
    '''
    Solves the system Ux=b by backwards elimination. U is a upper triangular matrix
    with with nonzero diagonal elements
    INPUT: U , array upper triangular matrix with nonzero diagonal elements.
           b, RHS vector
    OUTPUT : x, vector , solution to Ux=b
    '''
    try:
        n,m = U.shape
    except:
        print "Matrix must be two dimensional array."
        return
    if n != m:
        print "Matrix must be square"
        return
    elif n != len(b):
        print "Array must have same number of rows as column vector"
        return
    elif np.any(np.tril(U,k=-1)) or np.any(np.diag(U)==0.0):
        print "Array must be upper triangular with nonzero diagonal elements"
        return
    x = copy(b)
    for i in reversed(xrange(n)):
        for j in xrange(i+1,n):
            x[i] -= U[i,j]*x[j]
        x[i] /= U[i,i]
    return x
