#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 12 21:04:27 2017

@author: philippe
"""

import numpy as np

def Exp(x):
    if x > np.finfo(np.float64).max:
        return np.Inf
    elif x < np.finfo(np.float64).min:
        return 0.0
    elif x<0.0:
        return 1.0/Exp(-x)
    elif x<=2:
        A = list()
        B = list()
        A.append(1.0)
        A.append(1.0+x)
        B.append(1.0)
        B.append(1.0)
        n = 1
        error = x**2
        while abs(error) > np.finfo(np.float64).eps and n < 100:
            A.append( (n+1.0+x)*A[n+1] - n*x*A[n] )
            B.append( (n+1.0+x)*B[n+1] - n*x*B[n] )
            n+=1
            error *= x/n
        return A[-1]/B[-1]
    else:
        return Exp(x/2.0)**2   
 
    
def Exp2(x):
    if x > np.finfo(np.float64).max:
        return np.Inf
    elif x < np.finfo(np.float64).min:
        return 0.0
    elif x<0.0:
        return 1.0/Exp2(-x)
    elif x<2:        
        A = list()
        B = list()
        A.append(1.0)
        A.append(2.0+x)
        B.append(1.0)
        B.append(2.0-x)
        error = 2.0*x**3/(2.0-x)
        n = 2
        while abs(error) > np.finfo(np.float64).eps and n < 100:
            A.append( (4.0*n-2.0) * A[n-1] + x**2 * A[n-2] )
            B.append( (4.0*n-2.0) * B[n-1] + x**2 * B[n-2] )
            n+=1
            error = 2*(-1)**n * x**(2*n-1) / B[n-1]/B[n-2]
            print (error)
        return A[-1]/B[-1]
    else:
        return Exp2(x/2.0)**2   