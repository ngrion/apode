
# Agregar función que tabule datos
import numpy as np
import pandas as pd 
from numpy import random

# Generacion de distribuciones
# Ver: New code should use the xxx method of a default_rng() instance instead; see random-quick-start.
# rng = np.random.RandomState(10)  # deterministic random data
# rng.normal


def distribution_examples(dist,n,nbin=None):
    y = distribution_examples_aux(dist,n)
    df = pd.DataFrame({'x':y})
    if nbin==None:
        return df
    else:        
        return binning(df,nbin=nbin)

def distribution_examples_aux(dist,n):
    if dist=='uniform':
        mu = 100
        y = random.rand(n)*mu
    elif dist=='lognormal':
        y = 100*random.lognormal(mean=1.0, sigma=1.0, size=n)
    elif dist=='exponential':        
        y = 100*random.exponential(scale=2, size=n)
    elif dist=='pareto':             
        y = random.pareto(a=1, size=n)
    elif dist=='chisquare':  
        y = random.chisquare(df=2, size=n)
    elif dist=='gamma':          
        shape, scale = 2., 2.  # mean=4, std=2*sqrt(2)
        y = random.gamma(shape, scale, size=n)
    elif dist=='weibull':          
        a = 1.5 # shape
        y = 100*random.weibull(a, size=n)
    return y        

# generalizar columnanme
# puede remover nan
def binning(df,pos=0,nbin=None):
    if nbin==None:
        nbin = int(np.sqrt(df.shape[0]))
    s1 = df.groupby(pd.cut(df.iloc[:,pos], nbin)).count()
    s2 = df.groupby(pd.cut(df.iloc[:,pos], nbin)).mean()   
    dfb = pd.concat([s1, s2],axis=1 ).dropna()
    dfb.columns = ['weight','x']
    dfb.reset_index(drop=True,inplace=True)
    return dfb    