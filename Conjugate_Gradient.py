import numpy as np
from copy import copy, deepcopy
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
    bnorm =   np.linalg.norm(b)
    Anorm = np.linalg.norm(A)
    if M is None:
        p = copy(r)
        rsold = r.dot(r)
        Error = list()
        Error.append(np.sqrt(rsold)/(Anorm*np.linalg.norm(x0) + bnorm))
        for i in xrange(maxiter):
            Ap = A.dot(p)
            alpha = rsold/(p.dot(Ap))
            x0 += alpha*p
            r -= alpha*Ap
            rsnew = r.dot(r)
            Error.append(np.sqrt(rsnew)/(Anorm*np.linalg.norm(x0) + bnorm))
            if Error[i+1] < tol:
                break
            p  = r + (rsnew/rsold) * p
            rsold = rsnew
        return x0 , Error
    else:
        z = np.linalg.solve(M,r)
        p = copy(z)
        rsold = r.dot(z)
        Error = list()
        Error.append(np.linalg.norm(r)/(Anorm*np.linalg.norm(x0) + bnorm))
        for i in xrange(maxiter):
            Ap = A.dot(p)
            alpha = rsold/ p.dot(Ap)
            x0 += alpha*p
            r -= alpha*Ap
            Error.append(np.linalg.norm(r)/(Anorm*np.linalg.norm(x0) + bnorm))
            if Error[i+1] < tol:
                break
            z = np.linalg.solve(M,r)
            rsnew = r.dot(z)
            p = z + (rsnew/rsold)*p
            rsold = rsnew
        return x0, Error

def test_cg():
    n = 100
    np.random.seed(1240)
    b = np.random.randn(n)
    B =  np.random.rand(n,n)
    A = np.random.rand(n,n)
    A = A.dot(A.T)+np.diag([1.0*n for i in range(len(b))])
    y = cg(copy(A),copy(b),M = np.diag(np.diag(A)),tol=1e-10,maxiter=300)
    print np.linalg.norm(A.dot(y[0])-b)/(np.linalg.norm(A)*np.linalg.norm(y[0]) + np.linalg.norm(b))
    import matplotlib.pyplot as plt
    plt.plot(xrange(len(y[1])),-np.log10(y[1]))
    plt.ylim([-1,16])
    plt.show()

test_cg()
