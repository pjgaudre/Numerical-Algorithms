#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  9 16:56:58 2017

@author: philippe
"""
import numpy as np

def Wynn(N,E,SOFN):
    tiny = 1.0e-60
    huge = 1.0e60
    E.append(SOFN)
    if N == 0:
        return SOFN
    else:
        Aux2 = 0.0
        for j in reversed(range(1,N+1)):
            Aux1 = Aux2
            Aux2 = E[j-1]
            Diff= E[j]-Aux2
            if (abs(Diff)<tiny):
                E[j-1] = huge
            else:
                E[j-1] = Aux1 + 1.0/Diff
        if np.mod(N,2) == 0:
            return E[0]
        else:
            return E[1]

def main(a):
    N = len(a)
    S = 0 
    Q = []
    for n in range(N):
        S += a[n]
        Q.append(Wynn(n,Q[:],S))
    if np.mod(N,2) == 0:
        return Q[-1]
    else:
        return Q[N-2]
