"""Arnoldi algorithm."""
import numpy as np

def Arnoldi(A, q1, m):
    """Return the arnoldi iteration of A."""
    n, l = np.shape(A)
    if (n != l):
        print "Matrix must be square."
    if m+1 > n:
        T = n
    else:
        T = m+1
    Q = np.zeros((n, T))
    H = np.zeros((T, m))
    Q[:, 0] = q1
    for k in range(1, T):
        z = A.dot(Q[:, k-1])
        for i in range(k):
            H[i, k-1] = Q[:, i].dot(z)
            z -= H[i, k-1]*Q[:, i]
        H[k, k-1] = np.linalg.norm(z, 2)
        if H[k, k-1] < np.finfo(np.float64).eps:
            break
        Q[:, k] = z/H[k, k-1]
    if T == n:
        return (Q[:, :-1],H[:-1, :])
    else:
        return (Q, H)
