# ver valor de pline (no puede ser 1, por log)

import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt


from .inequality import gini_s,atkinson


def poverty_measure(y,method,*args):
    pline = args[0]
    args = args[1:]
 
    n = len(y)
    ys = np.sort(y)
    q = sum(ys<pline)    
    yp = ys[0:q] 
    # Foster–Greer–Thorbecke indices     
    if method == 'fgt0':
        p = q/n
    elif method == 'fgt1':
        br = (pline-yp)/pline
        p = sum(br)/n
    elif method == 'fgt2':
        br = np.power((pline-yp)/pline,2)
        p = sum(br)/n     
    elif method == 'fgt':        
        alpha = args[0]  # >= 0
        br = np.power((pline-yp)/pline,alpha)
        p = sum(br)/n  
    # Sen Index        
    elif method == 'sen':
        p0 = poverty_measure(y,'fgt0',pline)
        p1 = poverty_measure(y,'fgt1',pline)
        gp = gini_s(yp)
        p =  p0*gp + p1*(1-gp)    
    #  Sen-Shorrocks-Thon Index
    elif method == 'sst':
        p0 = poverty_measure(y,'fgt0',pline)
        p1 = poverty_measure(y,'fgt1',pline)
        gp = gini_s(yp)
        p =  p0*p1*(1+gp)        
    # Watts index (1968)
    elif method == 'watts':
        p = sum(np.log(pline/yp))/n
    # Indice de Clark, Ulph y Hemming (1981)        
    elif method== 'cuh':
        # existe c, valor por defecto?
        # 0<=alpha<=1
        if args is None:
            raise TypeError('Falta alpha')        
        alpha = args[0]
        if(alpha==0):
            p = 1 - np.power(np.product(yp/pline)/n,1/n)
        else:
            p = 1 - np.power((sum(np.power(yp/pline,alpha))+(n-q))/n,1/alpha)
    # Indice de Takayama            
    elif method=='takayama':
        u = (yp.sum()+(n-q)*pline)/n
        a = 0
        for i in range(0,q):
            a = a+(n-i+1)*y[i]
        for i in range(q,n):
            a = a+(n-i+1)*pline
        p = 1+1/n - (2/(u*n*n))*a              
    # Indice de Kakwani
    elif method=='kakwani':
        k = 2.0  # elegible
        a = 0.0
        u = 0.0
        for i in range(0,q):
            f = np.power(q-i+2,k) # ver +2
            a = a+f
            u = u+f*(pline-y[i])
        p = (q/(n*pline*a))*u            
    # Indice de Thon
    elif method=='thon':
        u = 0
        for i in range(0,q):
            u = u+(n-i+1)*(pline-y[i])
        p = (2/(n*(n+1)*pline))*u
    # Indice de Blackorby y Donaldson
    elif method=='bd':
        #ep = 2 # elegible
        ep = args[0]
        u = yp.sum()/q
        atkp = atkinson(yp,ep)
        yedep = u*(1-atkp)
        p = (q/n)*(pline-yedep)/pline        
    # Hagenaars
    elif method=='hagenaars':        
        ug =  np.exp(sum(np.log(yp))/q) # o normalizar con el maximo
        p = (q/n)*((np.log(pline)-np.log(ug))/np.log(pline))
    # Chakravarty (1983)
    elif method=='chakravarty':  
        b = args[0]
        #b = 0.5 # elegible  0<b<1
        p = sum(1- np.power(yp/pline,b))/n
    else:
        raise ValueError("Método "+ method + " no implementado.")  
    return p



# wheight, supone ordenados
# y es pandas, se puede multiplicar
def poverty_measure_w(ys,w,method,*args):
    pline = args[0]
    args = args[1:]
 
    n = sum(w)    
    q = sum(ys<pline)    
    yp = ys[0:q] 
    wp = w[0:q]
    # Foster–Greer–Thorbecke indices     
    if method == 'fgt0':
        p = sum(w[0:q])/n
    elif method == 'fgt1':
        br = ((pline-yp)/pline)*wp
        p = sum(br)/n
    elif method == 'fgt2':
        br = ((pline-yp)/pline)*wp
        br = np.power(br,2)
        p = sum(br)/n     
    elif method == 'fgt':        
        alpha = args[0]  # >= 0
        br = ((pline-yp)/pline)*wp
        br = np.power(br,alpha)
        p = sum(br)/n  
    else:
        raise ValueError("Método "+ method + " no implementado (datos agrupados).")
    return p        


# Datos ordenados
def tip_curve(ys,pline):
    n = len(ys)
    q = sum(ys<pline)   
    ygap = np.zeros(n)
    ygap[0:q] = (pline-ys[0:q])/pline
       
    z = np.cumsum(ygap)/n
    z = np.insert(z,0,0)
    p = np.arange(0,n+1)/n
    df = pd.DataFrame({'population':p,'variable':z})
    plt.plot(p,z)
    plt.title('TIP Curve')
    plt.ylabel('Cumulated poverty gaps')
    plt.xlabel('Cumulative % of population')
    plt.show()
    return df