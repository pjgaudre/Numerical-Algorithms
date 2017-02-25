import numpy as np
from copy import copy
from Thomas import Thomas
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def AdvectionDiffusion(N,M,Pe,T=1.0,L=1.0,Plot=False):
    '''
    AdvectionDiffusion solves the Advection Diffusion IBVP:
    u_t + u_x - Pe**(-1)u_xx = 0       in (0,L) , t > 0
    with the following initial and boundary conditions:
    u(0,x) = 0, u(t,0) = 1.0 , u_x(t,L) = 0.0.

    The equation is solved via an implicit, centered-in-space scheme.
    Given the number of sampling points in space "N", (step size h = L/N)
    and the number of sampling points in time "M", (time step k = T/M), the
    error in the approxiation is O(kh**2 + k**2). Additionally, the truncation error
    does contribute to a diffusion-like effect by a factor of O(Pe h**2 /6 - k/2).
    To cancel these effects, it is recommended to pick k = Pe*h**2 /3 or in terms of M and N
    M = ((3*T)/(Pe*L**2))*N**2. (Notice that optimally, we have M = O(N**2)...)

    Using the Thomas tridiagonal solver , the complexity of the this algorithm is O(MN)

    INPUT: N: integer, number of sampling points in space.
           M: integer, number of sampling points in time.
           Pe: Float, Peclet number
           T: Float, Length of time
           L: Float, Length in space.
           Plot: Bool, Should the solution be plotted.

    OUTPUT: U: array of dimension (M,N) , Solution to Advection Diffusion IBVP.
    '''
    h = L/N
    k = T/M
    u = np.zeros((M,N))
    KAPPA = 0.5*k/h
    GAMMA = k/Pe/h**2
    # -(KAPPA + GAMMA)u[i+1,j-1] + (1+2GAMMA)u[i+1,j] + (KAPPA - GAMMA)u[i+1,j+1] = u[i,j]
    # i = 0....M-1 , j = 1,....N
    a = np.full( N-2 , - KAPPA - GAMMA )
    b = np.full( N-1 , 1.0 + 2.0*GAMMA )
    c = np.full( N-2 , KAPPA - GAMMA )

    # INITIAL CONDITIONS
    # u(0,x) = 0.
    u[0,:] = 0.0
    # BOUNDARY CONDITIONS
    # u(t,0) = 1.0
    u[:,0] = 1.0
    # u_x(t,L) = 0.0
    #---> u[i+1,N+1] = u[i+1,N-1]
    #---> -2*GAMMA*u[i+1,N-1] + (1+2 GAMMA)*u[i+1,N] = u[i,N]
    a[-1] = -2*GAMMA

    for i in xrange(M-1):
        d = copy(u[i,1:])
        d[0] += KAPPA + GAMMA
        u[i+1,1:] = Thomas(copy(a),copy(b),copy(c),copy(d))

    if Plot:
        fig, ax = plt.subplots()
        x = np.linspace(0.0, L, N)
        line, = ax.plot(x, u[0,:])

        def animate(i):
            line.set_ydata(u[i,:])  # update the data
            return line,
        def init():
            line.set_ydata(np.ma.array(x, mask=True))
            return line,

        ani = animation.FuncAnimation(fig, animate, np.arange(1, M), init_func=init,
            interval=k, blit=True)
        plt.show()
    return u

def AdvectionDiffusion_test():
    Pe = 10.1
    N = 200
    L = 1.0
    T = 1.0
    M = int(np.ceil((3*T*N**2)/(Pe*L**2)))
    U = AdvectionDiffusion(N,M,Pe=Pe,T=3.0,L=1.0,Plot=True)

AdvectionDiffusion_test()
