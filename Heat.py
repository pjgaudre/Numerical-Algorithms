from scipy import sparse
import scipy.sparse.linalg as spl
from copy import copy
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
def Heat():
    '''
    The following function solves the homogenous Heat equation associated
    with Dirichlet and Neuman boundary conditions:
    u_t(x,y,t) = u_xx(x,y,t) + u_yy(x,y,t) , (x,y) in [0,1] x [0,1], t>0
    u(x,0,t) = 1.0  ,      u(x,1,t) = 0.0 ,    u_x(0,y,t) = u_x(1,y,t) = 0.0  t>0
    u(x,y,0) = 0.0 , (x,y) in (0,1) x (0,1),
    on a grid of N by N points using the method of finite differences:
    This is aconsistent scheme for approximation of the solution of this problem
    (called the theta-method)
    delta_{h}^{-1} u_{h,n} = theta * nabla_{h}^2 u_{h}^{n+1} +(1-theta) * nabla_{h}^2 u_{h}^{n}
    0<=theta<=1

    The resulting linear system is then solved using a Jacobi preconditioned sparse congugate gradient method.
    For theta = 0.5, the approximation is second order in both space and time.
    '''
    N=20
    k=0.01
    theta=0.5
    tol=1e-10
    h = 1./N
    GAMMA = k/h**2
    # Constructing Evolution matrices.
    # Resulting system is Anew u^n+1 = Aold u^n + c
    aold = (1.0 - theta)*GAMMA
    bold = 1.0 - 4.0*(1.0 - theta)*GAMMA
    anew = -theta*GAMMA
    bnew = 1.0 + 4.0*theta*GAMMA

    def HeatMat(a,b,N):
        offdiag = np.full( (N-1)*N , a )
        offdiag[0:(N-1)] *= 2.0
        T = np.diag([a]*(N-2),-1) + np.diag([b]*(N-1)) + np.diag([a]*(N-2),1)
        A = sparse.csr_matrix(np.kron(np.eye(N+1),T) + np.diag(offdiag[::-1],-(N-1))  + np.diag(offdiag,(N-1)))
        return A

    Aold = HeatMat(aold,bold,N)
    Anew = HeatMat(anew,bnew,N)
    c = np.zeros(Anew.shape[0])
    for i in xrange(N+1):
        c[i*(N-1)] = GAMMA

    fig = plt.figure()
    plt.title("Solution to heat equation")
    plt.xlabel("x")
    plt.ylabel("y")
    uold = np.zeros(Anew.shape[0])
    U = np.vstack((np.ones(N+1),np.reshape(uold, (N-1,N+1) ,order='F'),np.zeros(N+1)))
    ims = []
    im = plt.imshow(U, cmap='hot', interpolation='nearest',extent=(0.0,1.0,0.0,1.0), animated=True)
    plt.colorbar()
    ims.append([im])
    while True:
        unew = spl.cg(Anew,Aold.dot(uold[:]) + c, M = np.diag([bnew]*(N-1)*(N+1)),tol=tol)
        U = np.vstack((np.ones(N+1),np.reshape(unew[0], (N-1,N+1) ,order='F'),np.zeros(N+1)))
        im = plt.imshow(U, cmap='hot', interpolation='nearest',extent=(0.0,1.0,0.0,1.0), animated=True)
        #plt.colorbar()
        ims.append([im])
        if np.linalg.norm(unew[0]-uold,np.Inf)/np.linalg.norm(unew[0],np.Inf)<tol:
            break
        uold = copy(unew[0])
    ani = animation.ArtistAnimation(fig, ims, interval=50, blit=True, repeat_delay=1000)
    ani.save('Heat.mp4')
    return U

Heat()
