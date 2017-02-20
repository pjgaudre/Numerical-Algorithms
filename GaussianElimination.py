import numpy as np
from copy import copy, deepcopy

def pcgm(A,b,x,tol):
    r = b-MatMul(A,x)
    z = solve(np.diag(np.diagonal(A)),r)
    p = z
    rsold = r.dot(z)
    Error = []
    er = np.linalg.norm(r)
    Error.append(er)
    for i in xrange(len(b)):
        Ap = MatMul(A,p)
        alpha = rsold/ p.dot(Ap)
        x += alpha*p
        r -= alpha*Ap
        er = np.linalg.norm(r)
        Error.append(er)
        if er < tol:
            break
        z = solve(np.diag(np.diagonal(A)),r)
        rsnew = r.dot(z)
        p = z + (rsnew/rsold)*p
        rsold = rsnew
    return x,Error

def cgm(A,b,x,tol):
    r = b-MatMul(A,x)
    p = r
    rsold = r.dot(r)
    Error = []
    for i in xrange(len(b)):
        Error.append(np.sqrt(rsold))
        Ap = A.dot(p)
        alpha = rsold/(p.dot(Ap))
        x += alpha*p
        r -= alpha*Ap
        rsnew = r.dot(r)
        if np.sqrt(rsnew)<tol:
            break
        p  = r + (rsnew/rsold) * p
        rsold = rsnew
    return x, Error

def solve(M,r):
    return np.linalg.solve(M,r)

def MatMul(B,q):
    return B.dot(q)

def SOR(A,b,w = 1.0,NumIter=200,tol=np.finfo(np.float64).eps):
    n,m = A.shape
    k = len(b)
    tol *= np.linalg.norm(b,np.inf)

    if n != m:
        print "Matrix must be square"
    elif k != n:
        print "column vector must have same number of rows as matrix."
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
        if Error[k] < tol:
            break
        x0 = copy(x)
    return x, Error

def SSOR(A,b,w = 1.0,NumIter=500,tol=np.finfo(np.float64).eps):
    n = len(b)
    x0 = np.array([0.0 for i in xrange(n)])
    xhalf = copy(x0)
    x =  copy(x0)
    Error = []
    for k in xrange(NumIter):

        for i in xrange(n):
            sigma = 0.0
            for j in xrange(i):
                sigma += A[i,j]*xhalf[j]
            for j in xrange(i+1,n):
                sigma += A[i,j]*x0[j]
            sigma = (b[i]-sigma)/A[i,i]
            xhalf[i] = x0[i] + w*(sigma-x0[i])

        for i in reversed(xrange(n)):
            sigma = 0.0
            for j in xrange(i):
                sigma += A[i,j]*xhalf[j]
            for j in xrange(i+1,n):
                sigma += A[i,j]*x[j]
            x[i] = xhalf[i] + w*(sigma-xhalf[i])

        Error.append(np.linalg.norm(x - x0))

        if Error[k] < tol:
            break
        x0 = copy(x)
    return x, Error



np.random.seed(100)
n = 100
b = np.random.rand(n)
A = np.random.rand(n,n)
A = (A + A.T)/2.0
A = A.dot(A.T)+np.diag([1.0*n for i in range(len(b))])
#M = np.diag(np.diag(A)**(-1))
x = np.zeros(n)
#x
x,Error1 = pcgm(A,b,x,1e-15)
#x = np.zeros(n)
#x,Error2 = cgm(A,b,x,1e-15)
#import matplotlib.pyplot as plt
plt.plot(range(len(Error1)),np.log10(Error1))
#plt.plot(range(len(Error2)),np.log10(Error2))
#plt.show()
#
#np.linalg.norm(A.dot(x)-b)
E
x,E = SOR(A,b,w=0.25)
import matplotlib.pyplot as plt
plt.plot(range(len(E)),np.log10(E))
plt.show()


np.linalg.norm(A,np.Inf)
np.linalg.norm(b,np.Inf)
