import numpy as np
from copy import copy, deepcopy
import matplotlib.pyplot as plt
import seaborn as se

def cg(A, b, x0=None, tol=1e-05, maxiter=None, M=None):
    '''
    Use Conjugate Gradient iteration to solve Ax = b
    INPUT: A : Symmetric positive definite dense matrix
           b : RHS vector
    OUTPUT: x : solution to Ax = b
            Error: Vector containing the Normwise Backward Error estimate at each iteration.
    OTHER PARAMETERS: x0:, vector, Initial estimate of x.
                      tol: float, Tolerance to achieve. The algorithm terminates
                           when the Normwise Backward Error estimate is below tol or
                           when the maximum number of iterations has been reached.
                      maxiter: Maximum number of iterations. Iteration will stop
                              after maxiter steps even if the specified
                              tolerance has not been achieved.
                     M: Preconditioner for A. The preconditioner should approximate
                        the inverse of A. Effective preconditioning dramatically
                        improves the rate of convergence, which implies that
                        fewer iterations are needed to reach a given error tolerance.
    '''
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
