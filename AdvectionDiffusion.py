import numpy as np
from copy import copy
from Thomas import Thomas
import matplotlib.pyplot as plt
def AdvectionDiffusion(N=5,Pe=10.0):
    h = 1.0/N
    k = Pe*(h**2)/3.0
    M = int(np.ceil(1.0/k))
    gamma = 1.0/3.0
    C = Pe*h/6.0
    a = - C - gamma
    b = 1.0+2.0*gamma # 5.0/3.0
    c = C - gamma
    a = np.full(N-2,a)
    a[-1] = -2*gamma
    b = np.full(N-1,b)
    c = np.full(N-2,c)
    # Initial conditions and  boundary conditions
    u = np.zeros((M,N))
    u[0,:] = np.zeros(N)
    RHSvec = np.zeros(N-1)
    RHSvec[0] = -C
    u[:,0] = np.ones(M)
    for i in xrange(M-1):
        u[i+1,1:] = Thomas(copy(a),copy(b),copy(c),copy(u[i,1:]+RHSvec))
    return u


N = 100
u = AdvectionDiffusion(N)
x = np.linspace(0,1,N)

u.shape

for i in xrange(u.shape[0]):
    plt.plot(x,u[i,:])
plt.show()
