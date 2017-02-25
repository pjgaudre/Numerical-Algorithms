import numpy as np
from copy import copy
from Thomas import Thomas
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def AdvectionDiffusion(N,M,Pe,T=1.0,L=1.0,Plot=False):
    '''

    '''
    h = L/N
    k = T/M
    u = np.zeros((M,N))
    KAPPA = 0.5*k/h
    GAMMA = k/Pe/h**2
    a = np.full( N-2 , -(KAPPA + GAMMA) )
    b = np.full( N-1 , 1.0 + 2.0*GAMMA )
    c = np.full( N-2 , KAPPA - GAMMA )

    # INITIAL CONDITIONS
    # u(0,x) = 0.
    u[0,:] = 0.0
    # BOUNDARY CONDITIONS
    # u(t,0) = 1.0
    u[:,0] = 1.0
    # u_x(t,L) = 0.0 ---> (1+2 GAMMA)*u[i+1,N] - 2*GAMMA*u[i+1,N-1] = u[i,N]
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
        #Init only required for blitting to give a clean slate.
        def init():
            line.set_ydata(np.ma.array(x, mask=True))
            return line,

        ani = animation.FuncAnimation(fig, animate, np.arange(1, M), init_func=init,
            interval=k, blit=True)
        plt.show()
    return u

def AdvectionDiffusion_test():
    Pe = 10.1
    N = 100
    T = 3.0
    k = Pe/3.0/N**2
    M = int(np.ceil(T/k))
    U = AdvectionDiffusion(N,M,Pe=Pe,T=3.0,L=1.0,Plot=True)

AdvectionDiffusion_test()
