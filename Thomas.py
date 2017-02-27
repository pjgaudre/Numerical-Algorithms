import numpy as np
from numpy.random import rand, seed
from copy import copy
from numpy import diag

def Thomas(a,b,c,d):
    '''
    x = Thomas(a,b,c,d) returns the solution of the tridiagonal system of equations:
    Tx = d, where T = diag(a,-1) + diag(b) + diag(c,1).
    '''
    n = len(d)
    for i in xrange(1,n):
        m = a[i-1]/b[i-1]
        b[i] -= m*c[i-1]
        d[i] -= m*d[i-1]
    x = b
    x[-1] = d[-1]/b[-1]
    for i in xrange(n-2,-1,-1):
        x[i] = (d[i]-c[i]*x[i+1])/b[i]
    return x
