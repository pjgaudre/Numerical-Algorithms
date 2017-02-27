import numpy as np
from copy import copy
import matplotlib.pyplot as plt
import seaborn as se

def SOR(A,b,w = 1.0,NumIter=200,tol=np.finfo(np.float64).eps):
    try:
        n,m = A.shape
    except:
        print "Matrix must be two dimensional array."
        return
    bnorm = np.linalg.norm(b,np.inf)
    Anorm = np.linalg.norm(A,np.inf)
    if n != m:
        print "Matrix must be square"
        return
    elif len(b) != n:
        print "column vector must have same number of rows as matrix."
        return
    x0 = np.array([0.0 for i in xrange(n)])
    x =  copy(x0)
    Error = []
    for k in xrange(NumIter):
        for i in xrange(n):
            sigma = 0.0
            for j in xrange(i):
                sigma += A[i,j]*x[j]
            for j in xrange(i+1,n):
                sigma += A[i,j]*x0[j]
            sigma = (b[i]-sigma)/A[i,i]
            x[i] = x0[i] + w*(sigma-x0[i])
        Error.append(np.linalg.norm(x - x0,np.inf))
        if Error[k] < tol*(Anorm*np.linalg.norm(x,np.inf)+bnorm) :
            break
        x0 = copy(x)
    return x, Error
