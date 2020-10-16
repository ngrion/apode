
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .poverty import poverty_measure,poverty_measure_w,tip_curve
from .inequality import ineq_measure,ineq_measure_w,lorenz_curve,pen_parade
from .welfare import welfare_measure,welfare_measure_w
from .polarization import polarization_measure
from .concentration import concentration_measure
from .distributions import distribution_examples,binning,default_rng
from .util import eval_measure_list,joinpar


# Temas pendientes
#  - controlar que w y wheigth sean numericos
#  - completar funcion describe
#  - tratamiento de nan?
#  - modifficar elementos del objetos de manera segura
class IneqMeasure:
    """
    Measuring Inequality, Poverty, Welfare, Polarization and Concentration.
    """    
    def __init__(self, data, varx=None, weight=None, issorted=False):
        self._data = data
        self._varx = varx
        self._weight = weight
        self._issorted = issorted
        self._validate = self._fvalidate()
    
    def _fvalidate(self):
        if not isinstance(self._data, pd.DataFrame):            
            try:          
                self._data = pd.DataFrame(self._data)  
            except ValueError:
                print("No se puede convertir a un DataFrame")
        if self._varx==None:
            if self._weight==None:
                if self._data.shape[1]==1:            
                    self._varx = self._data.columns[0]
                else: 
                    raise ValueError("Se requiere el nombre de la variable (cols>1).")
            else:        
                raise ValueError("Se requiere el nombre de la variable.")
        elif not (self._varx in self._data.columns):
            raise ValueError("El nombre de la variable no existe.")
        if not(self._weight==None):          
            if not (self._weight in self._data.columns):
                raise ValueError("El nombre de la variable frecuencia no existe.")
        if not isinstance(self._issorted, bool):
            raise ValueError("El argumento issorted debe ser bool.")
        # tipos de datos dataframe (x y weight)
        if not pd.api.types.is_numeric_dtype(self._data[self._varx]):
            raise ValueError("La variable debe ser numérica.")
        if not(self._weight==None): 
            if not pd.api.types.is_numeric_dtype(self._data[self._weight]):
                raise ValueError("La variable frecuencia debe ser numérica.")

        return True

    # modificar valores 
    @property
    def varx(self):
        return self._varx
        
    @varx.setter
    def varx(self, varx):
        if not (varx in self._data.columns):
            raise ValueError("El nombre de la variable no existe.")
        else:
            self._varx = varx

    @property
    def weight(self):
        return self._weight
        
    @weight.setter
    def weight(self, weight):
        if not (weight in self._data.columns):
            raise ValueError("El nombre de la variable frecuencia no existe.")
        else:
            self._weight = weight

    @property
    def issorted(self):
        return self._issorted
        
    @issorted.setter
    def issorted(self, issorted):
        if not isinstance(issorted, bool):
            raise ValueError("El argumento issorted debe ser bool.")
        else:
            self._issorted = issorted

    @property
    def data(self):
        return self._data
        
    @data.setter
    def data(self, data):
        df_tmp = self._data.copy()
        try:
            self._data = data
            self._validate = self._fvalidate()
        except:
            self._data = df_tmp # revierte modificacion
            raise ValueError("El dataframe no es válido")


    # Métodos sobre el dataframe
    def describe(self):
        return _describe_apode(self._data,self._varx ,self._weight,self._issorted)

    def columns(self):
        return self._data.columns.values.tolist()

    def ndim(self):
        return self._data.ndim

    def shape(self):
        return self._data.shape

    def size(self):
        return self._data.size        

    def display(self):
        display(self._data) 


    # Metodos sobre los indicadores
    def _fsort(self):
        self._data = self._data.sort_values(by=self._varx)
        self._issorted = True        

    def sort(self):
        self._fsort()

    def _get_arg(self,sort):
        # ordena dataframe si es requerido
        if sort and (not self._issorted):
            self._fsort()
        y = self._data[self._varx].values    
        if self._weight==None:
            w = None
            dow = False
        else:
            w = self._data[self._weight].values
            dow = True        
        return dow,y,w
            
    # no modifica orden en _data        
    def _get_arg_groupby(self,group,sort):
        # ordena dataframe si es requerido
        if sort:
            group = group.sort_values(by=self._varx)
        y = group[self._varx].values    
        if self._weight==None:            
            w = None
            count = group.shape[0]
            dow = False            
        else:
            w = group[self._weight].values
            count = sum(w)
            dow = True               
        return dow,y,w,count


        
    # tratamiento general de grupos    
    def _measure_aux(self,fmeasure, method, gby,*args):
        if gby==None:
            dow,ys,w = self._get_arg(sort=True) 
            return fmeasure(method,dow,ys,w,*args)
        else:
            if not (gby in self._data.columns):
                raise ValueError("El nombre de la variable no existe.")
            grouped = self._data.groupby(gby)
            a = []; b = []; c = []
            for name, group in grouped:
                dow,ys,w,count = self._get_arg_groupby(group,sort=True)
                a.append(name)
                b.append(fmeasure(method,dow,ys,w,*args))
                c.append(count)   
            xname =  self._varx + "_measure"            
            wname = self._varx + "_weight"
            return pd.DataFrame({xname:b,wname:c},index=pd.Index(a))         

    def _poverty_aux(self,method,dow,ys,w,*args):             
        if dow:
            return poverty_measure_w(ys,w,method,*args)
        else:
            return poverty_measure(ys,method,*args)  

    def poverty(self, method,*args,gby=None): 
        return self._measure_aux(self._poverty_aux, method, gby,*args)  


    def _ineq_aux(self,method,dow,ys,w,*args):  
        if dow:
            return ineq_measure_w(ys,w,method,*args)
        else:
            return ineq_measure(ys,method,*args)

    def ineq(self, method,*args,gby=None): 
        return self._measure_aux(self._ineq_aux, method, gby,*args)  


    def _welfare_aux(self,method,dow,ys,w,*args): 
        if dow:
            return welfare_measure_w(ys,w,method,*args)
        else:
            return welfare_measure(ys,method,*args)   
        
    def welfare(self, method,*args,gby=None): 
        return self._measure_aux(self._welfare_aux, method, gby,*args) 


    def _polar_aux(self,method,dow,ys,w,*args): 
        if dow:
            return polarization_measure_w(ys,w,method,*args)
        else:
            return polarization_measure(ys,method,*args)   
        
    def polar(self, method,*args,gby=None): 
        return self._measure_aux(self._polar_aux, method, gby,*args)


    def _conc_aux(self,method,dow,ys,w,*args): 
        if dow:
            return concentration_measure_w(ys,w,method,*args)
        else:
            return concentration_measure(ys,method,*args)   
        
    def conc(self, method,*args,gby=None): 
        return self._measure_aux(self._conc_aux, method, gby,*args)

    # graficos
    def tip(self,*args,**kwargs):    # Curve TIP
        dow,ys,w = self._get_arg(sort=True)
        if dow:
            raise ValueError("No implementado (datos agrupados).")
        else:
            return tip_curve(ys,*args,**kwargs)        

    def lorenz(self,*args,**kwargs):    # Curve de Lorenz
        dow,ys,w = self._get_arg(sort=True)
        if dow:
            raise ValueError("No implementado (datos agrupados).")
        else:
            return lorenz_curve(ys,*args,**kwargs)              
        
    def pen(self,*args,**kwargs):    # Pen's Parade
        dow,ys,w = self._get_arg(sort=True)
        if dow:
            raise ValueError("No implementado (datos agrupados).")
        else:
            return pen_parade(ys,*args,**kwargs)  

# Falta implementar percentiles en caso agrupado
def _describe_apode(df,x,w,issort):
    bins = df.shape[0]
    if w==None:            
        v = [False,bins,issort]
        ind = pd.Index(['weight','bins','sorted'])   
        dfd1 = pd.DataFrame(data=v,index=ind,columns=[x])
        dfd2 = pd.DataFrame(df[x].describe())
        return pd.concat([dfd1,dfd2])
    else: 
        n = sum(df[w])    
        mu = sum(df[x]*df[w])/n
        v = [True,bins,issort,n,mu,min(df[x]),max(df[x])]
        ind = pd.Index(['weight','bins','sorted','count','mean','min','max'])
        return pd.DataFrame(data=v,index=ind,columns=[x])        