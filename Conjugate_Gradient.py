import numpy as np
from copy import copy, deepcopy
import matplotlib.pyplot as plt
import seaborn as se

def cg(A, b, x0=None, tol=1e-05, maxiter=None, M=None):
    try:
         np.linalg.cholesky(A)
         n,m = A.shape
    except:
        print "Matrix must be square, symmetric and positive definite."
        return
    if n != len(b):
        print "Output vector must have same number of rows as matrix."
        return
    if maxiter is None:
        maxiter = n
    if x0 is None:
        r = copy(b)
        x0 = np.array([0.0]*n)
    else:
        r = b - A.dot(x0)
    bnorm =   np.linalg.norm(b,2)
    Anorm = np.linalg.norm(A,'fro')
    if M is None:
        p = copy(r)
        rsold = r.dot(r)
        Error = list()
        for i in xrange(maxiter):
            Ap = A.dot(p)
            alpha = rsold/(p.dot(Ap))
            x0 += alpha*p
            r -= alpha*Ap
            rsnew = r.dot(r)
            Error.append(np.sqrt(rsnew)/(Anorm*np.linalg.norm(x0,2) + bnorm))
            if Error[i] < tol:
                break
            p  = r + (rsnew/rsold) * p
            rsold = rsnew
        return x0 , Error
    else:
        z = np.linalg.solve(M,r)
        p = copy(z)
        rsold = r.dot(z)
        Error = list()
        for i in xrange(maxiter):
            Ap = A.dot(p)
            alpha = rsold/ p.dot(Ap)
            x0 += alpha*p
            r -= alpha*Ap
            Error.append(np.linalg.norm(r)/(Anorm*np.linalg.norm(x0,2) + bnorm))
            if Error[i] < tol:
                break
            z = np.linalg.solve(M,r)
            rsnew = r.dot(z)
            p = z + (rsnew/rsold)*p
            rsold = rsnew
        return x0, Error

def test_cg():
    n = 24
    np.random.seed(10)
    b = np.random.randn(n)
    A = np.random.randn(n,n)+12*np.random.rand(n,n) -13*np.random.rand(n,n)
    A = (A.T + A)/2.0
    A = A.dot(A.T)+np.diag([1.0 for i in range(n)])
    y = cg(copy(A),copy(b),M = np.diag(np.diag(A)),tol=1e-14,maxiter=300)
    z = cg(copy(A),copy(b),tol=1e-14,maxiter=300)
    se.plt.plot(xrange(1,len(y[1])+1),np.log10(y[1]),'-ro',label="Jacobi Preconditioned CG")
    se.plt.plot(xrange(1,len(z[1])+1),np.log10(z[1]),'--bs',label="CG")
    se.plt.ylim([-16,1])
    se.plt.xlim([0,len(y[1])+1])
    se.plt.ylabel("Log10 of Normwise Backward Error")
    se.plt.xlabel("Number of Iterations")
    se.plt.title("Error analysis of CG method")
    se.plt.xticks(xrange(1,len(y[1])+1))
    se.plt.legend()
    se.plt.show()

test_cg()
