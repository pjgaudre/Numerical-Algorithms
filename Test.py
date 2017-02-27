import numpy as np
from copy import copy, deepcopy
import matplotlib.pyplot as plt
import seaborn as se

from AdvectionDiffusion import *
from Cholesky import *
from Conjugate_Gradient import *
from Elementary_Functions import *
from Forward import *
from LU_LUP import *
from SOR import *
from Thomas import *

def AdvectionDiffusion_test():
    Pe = 0.1
    M = 30000
    L = 2.0
    T = 3.0
    N = int(np.ceil(np.sqrt(Pe*M/3./T)*L))
    U = AdvectionDiffusion(N,M,Pe=Pe,T=T,L=L,Plot=True)

def Cholesky_test():
    n=40
    np.random.seed(10)
    # Two dimensional test
    A = np.random.rand(n)
    L = Cholesky(copy(A))
    # Non square matrix case
    A = np.random.rand(n,n+1)
    L = Cholesky(copy(A))
    # Non symmetric case
    A = np.random.rand(n,n)
    L = Cholesky(copy(A))
    # When it works, does it converge?
    A = A.dot(A.T)
    L = Cholesky(copy(A))
    if np.linalg.norm(L.dot(L.T)-A)<=1e-10:
        print "Test passed"
    else:
        print "test failed"
    # Matrix is not positive definite
    A = A - np.diag([10.0]*n)
    L = Cholesky(copy(A))

def test_cg():
    n = 24
    np.random.seed(10)
    b = np.random.randn(n)
    A = np.diag(5.*np.ones(n)) + np.diag(2.*np.ones(n-1),1) + np.diag(2.*np.ones(n-1),-1)+ np.diag(np.ones(n-2),2) + np.diag(np.ones(n-2),-2)
    y = cg(copy(A),copy(b),M = np.diag(np.diagonal(A)),tol=1e-14,maxiter=300)
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


def Test_Exp():
    Error = array([ abs((Exp(x)-exp(x))/exp(x)) for x in linspace(-709,709,100000) ])
    if max(Error)<1e-12 and Exp(-800) == 0.0 and Exp(800) == Inf :
        print("All test passed!")
    else:
        print("Test Failed")

def Test_Log():
    Error = array([ abs((Log(x)-log(x))/log(x)) for x in linspace(1e-10,1e20,100000) ])
    if max(Error)<1e-12 and Log(0.0) == -Inf and Log(1e600) == Inf :
        print("All test passed!")
    else:
        print("Test Failed")

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

def LU_test():
    n = 10
    A = np.random.randn(n,n)
    A = (A.T + A)/2.0
    L,U = LU(A)
    if np.linalg.norm(L.dot(U)-A,'fro') <1.e-14:
        print "test passed"
    else:
        print "test failed"

def LUP_test():
    n = 10
    A = np.random.randn(n,n)
    L,U,P = LUP(A)
    if np.linalg.norm(L.dot(U)-A[P,:],'fro') <1.e-14:
        print "test passed"
    else:
        print "test failed"


def test_SOR():
    n = 24
    np.random.seed(10)
    b = np.random.randn(n)
    A = np.diag(7.*np.ones(n)) + np.diag(2.*np.ones(n-1),1) + np.diag(2.*np.ones(n-1),-1)+ np.diag(np.ones(n-2),2) + np.diag(np.ones(n-2),-2)
    y1 = SOR(A,b,w = 1.0,NumIter=200,tol=np.finfo(np.float64).eps)
    y2 = SOR(A,b,w = 1.5,NumIter=200,tol=np.finfo(np.float64).eps)
    y3 = SOR(A,b,w = 0.5,NumIter=200,tol=np.finfo(np.float64).eps)
    se.plt.plot(xrange(1,len(y1[1])+1),np.log10(y1[1]),'--b',label="w=1.0")
    se.plt.plot(xrange(1,len(y2[1])+1),np.log10(y2[1]),'--g',label="w=1.5")
    se.plt.plot(xrange(1,len(y3[1])+1),np.log10(y3[1]),'--k',label="w=0.5")
    se.plt.ylim([-16,1])
    se.plt.ylabel("Log10 of Normwise Backward Error")
    se.plt.xlabel("Number of Iterations")
    se.plt.title("Error analysis of SOR method")
    se.plt.legend()
    se.plt.show()


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
