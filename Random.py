import numpy as np
from copy import copy, deepcopy

def cg(A, b, x0=None, tol=1e-05, maxiter=None, M=None):
    if maxiter is None:
        maxiter = len(b)
    if x0 is None:
        r = b
        x0 = np.array([0.0]*len(b))
    else:
        r = b - MatMul(A,x0)
    if M is None:
        p = r
        rsold = r.dot(r)
        Error = []
        Error.append(np.sqrt(rsold))
        for i in xrange(maxiter):
            Ap = MatMul(A,p)
            alpha = rsold/(p.dot(Ap))
            x0 += alpha*p
            r -= alpha*Ap
            rsnew = r.dot(r)
            Error.append(np.sqrt(rsnew))
            if Error[i+1] < tol:
                break
            p  = r + (rsnew/rsold) * p
            rsold = rsnew
        return x0 , Error
    else:
        z = solve(M,r)
        p = z
        rsold = r.dot(z)
        Error = []
        Error.append(np.linalg.norm(r))
        for i in xrange(maxiter):
            Ap = MatMul(A,p)
            alpha = rsold/ p.dot(Ap)
            x0 += alpha*p
            r -= alpha*Ap
            Error.append( np.linalg.norm(r) )
            if Error[i+1] < tol:
                break
            z = solve(M,r)
            rsnew = r.dot(z)
            p = z + (rsnew/rsold)*p
            rsold = rsnew
        return x0, Error


def pcgm(A,b,M,x,tol):
    r = b-MatMul(A,x)
    z = solve(M,r)
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
        z = solve(M,r)
        rsnew = r.dot(z)
        p = z + (rsnew/rsold)*p
        rsold = rsnew
    return x , Error

def cgm(A,b,x,tol):
    r = b-MatMul(A,x)
    p = r[:]
    rsold = r.dot(r)
    delta_old = 0.0
    gamma_old = 0.0
    for i in xrange(len(b)):
        Ap = MatMul(A,p)
        gamma_new = rsold/(p.dot(Ap))
        x += gamma*p
        r -= gamma*Ap
        rsnew = r.dot(r)
        delta_new = rsnew/rsold
        p  = r + delta_new * p
        alpha = 1.0/gamma_new + delta_old/gamma_old ????
        beta2_old = delta_new/gamma_new**2
        if k = 1:
            c1 = 1.0
            DELTA_old = alpha
        else:
            omega = np.sqrt((DELTA_old-alpha)**2+4*b)

        rsold = rsnew
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


xsol = cg(A,b,x0 =np.zeros(n),M = np.diag(np.diag(A)))

np.linalg.norm(A.dot(xsol[0])-b)

np.random.seed(100)
n = 100
b = np.random.rand(n)
A = np.random.rand(n,n)
A = (A + A.T)/2.0
A = A.dot(A.T)+np.diag([1.0*n for i in range(len(b))])
#
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
