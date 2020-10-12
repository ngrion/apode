
import numpy as np
from .inequality import gini_s


def polarization_measure(y,method,*args):    
    if method == 'er':
        p = polarization_er(y)
    elif method == 'wlf':
        p = polarization_wlf(y)
    else:
        raise ValueError("Método "+ method + " no implementado.")  
    return p


def polarization_measure_w(y,w,method,*args):    
    raise ValueError("Método "+ method + " no implementado (datos agrupados).")
    p = []
    return p


# Esteban and Ray index of polarization
# generalizar parametro
def polarization_er(y):
    n = len(y)
    alpha = 1 # (0,1.6]
    p_er = 0
    for i in range(0,n):
        for j in range(0,n):
            pi = 1/n
            pj = 1/n
            p_er = p_er + np.power(pi,1+alpha)*pj*abs(y[i]-y[j])
    return p_er 


# Wolfson index of bipolarization
# ver que n> sea grande
def polarization_wlf(y):
    ys = np.sort(y)
    ysa = np.cumsum(ys)/sum(ys)
    n = len(y)
    if (n % 2) == 0:
        i = int(n/2)
        L = (ysa[i-1]+ysa[i])/2
    else:
        i = int((n+1)/2)
        L = ysa[i-1]
    g = gini_s(ys)
    p_w = (np.mean(y)/np.median(y))*(0.5-L-g)
    return p_w