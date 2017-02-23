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

def Thomas_test():
    n = 50
    a = rand(n-1)
    b = rand(n)
    c = rand(n-1)
    d = rand(n)
    A = diag(a,-1) + diag(b) + diag(c,1)
    Error = np.linalg.norm(Thomas(a, b, c, copy(d)) - np.linalg.solve(A,copy(d)))
    if Error<=1e-10:
        print "Test passed"
    else:
        print "Test failed"
Thomas_test()
