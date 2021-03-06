#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 11 16:18:38 2017

@author: philippe
"""

from numpy import exp,finfo,float64,e,Inf,array,log,linspace

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
        return Inf
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
        numItr = 1
        while abs(error) > finfo(float64).eps and numItr <= 500:
            Atemp = bn * A2 + an * A1
            Btemp = bn * B2 + an * B1
            error *= an*B1/Btemp
            A1 = A2
            A2 = Atemp
            B1 = B2
            B2 = Btemp
            bn += 4.0
            numItr += 1
        return A2/B2
    else:
        return Exp(0.5*x)**2.0

def Log(x):
    ''' Returns the value of the natural logarithm log base e ≈ 2.71828...
     being euler's number. This function relies on the convergent power series
     expansion for x>0:
     log(x) = 2*( z + 1/3 z**3 + 1/5 z**5 +... ) where z = (x-1)/(x+1) to obtain
     a good starting point estimate to use the Newton-Raphson method.
     We also use the identies log(x) = log(e*x)-1.0 and log(x) = -log(1/x)
     to reduce the domain of approximation to 1.0/e<=x<e.
    '''
    if x<0:
        print ("x must be a postive real number.")
    elif x>finfo(float64).max:
        return Inf
    elif x < 1.0/finfo(float64).max:
        return -Inf
    elif x<1.0/e:
        return Log(e*x)-1.0
    elif 1.0/e<=x and x<e:
        z = (x-1.0)/(x+1.0)
        yold = z
        ynew = z**2
        for n in xrange(5):
            z *= ynew*(n+0.5)/(n+1.5)
            yold += z
        yold *= 2
        z = x/exp(yold) - 1.0
        ynew = yold + z
        numItr = 1
        while abs(z) > 2*finfo(float64).eps and numItr <= 500:
            yold = ynew
            z = x/exp(yold) - 1.0
            ynew = yold + z
            numItr += 1
        return ynew
    else:
        return -Log(1.0/x)
