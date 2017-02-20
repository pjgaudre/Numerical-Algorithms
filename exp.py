#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 11 16:18:38 2017

@author: philippe
"""

import numpy as np

def Exp(x):
    ''' Returns the value of the exponential function e^x with e ≈ 2.71828...
     being euler's number. This function relies on the generalized continued fraction:
    exp(x) = 1 + 2x/
                 2-x + x**2/
                       6 + x**2/
                           10 + x**2/
                                14 + ...
    In addition to the identies exp(x) = 1/exp(-x) and exp(x) = exp(0.5*x)**2.
    This function agrees in relative error with the one from the numpy package
    for up to 12 correct digits.
    '''
    if x > 709.78271289338397:
        return np.Inf
    elif x < -709.78271289338397:
        return 0.0
    elif x<0.0:
        return 1.0/Exp(-x)
    elif x<2.0:
        A1 = 1.0
        A2 = 2.0+x
        B1 = 1.0
        B2 = 2.0-x
        an = x**2
        bn = 6.0
        error = 2.0*x*an/B2
        while abs(error) > np.finfo(np.float64).eps:
            Atemp = bn * A2 + an * A1
            Btemp = bn * B2 + an * B1
            error *= an*B1/Btemp
            A1 = A2
            A2 = Atemp
            B1 = B2
            B2 = Btemp
            bn += 4.0
        return A2/B2
    else:
        return Exp(0.5*x)**2.0

def Test_Exp():
    x = np.linspace(-709,709,100000)
    Error = np.array([ abs((Exp(y)-np.exp(y))/np.exp(y)) for y in x ])
    if max(Error)<1e-12 and Exp(-800) == 0.0 and Exp(800) == np.Inf :
        print("All test passed!")
    else:
        print("Test Failed")

Test_Exp()
