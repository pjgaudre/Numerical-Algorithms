from scipy import sparse
import scipy.sparse.linalg as spl
from copy import copy
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def Heat():
    '''
    The following function solves the homogenous Heat equation associated
    with Dirichlet and Neuman boundary conditions on a grid of N by N points using the
    method of finite differences. The resulting linear system is then solved
    using a preconditioned congugate gradient method.

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
    uold = np.zeros(Anew.shape[0])
    U = np.vstack((np.ones(N+1),np.reshape(uold, (N-1,N+1) ,order='F'),np.zeros(N+1)))
    ims = []
    im = plt.imshow(U, cmap='hot', interpolation='nearest', animated=True)
    ims.append([im])
    while True:
        unew = spl.cg(Anew,Aold.dot(uold[:]) + c, M = np.diag([bnew]*(N-1)*(N+1)),tol=tol)
        U = np.vstack((np.ones(N+1),np.reshape(unew[0], (N-1,N+1) ,order='F'),np.zeros(N+1)))
        im = plt.imshow(U, cmap='hot', interpolation='nearest', animated=True)
        ims.append([im])
        if np.linalg.norm(unew[0]-uold,np.Inf)/np.linalg.norm(unew[0],np.Inf)<tol:
            break
        uold = copy(unew[0])
    ani = animation.ArtistAnimation(fig, ims, interval=1, blit=True, repeat_delay=1000)
    ani.save('Heat.mp4')
    plt.show()
    return U

Heat()
