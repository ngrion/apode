
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .poverty import poverty_measure,poverty_measure_w,tip_curve
from .inequality import ineq_measure,ineq_measure_w,lorenz_curve
from .welfare import welfare_measure,welfare_measure_w
from .polarization import polarization_measure
from .concentration import concentration_measure
from .distributions import distribution_examples,binning
from .util import eval_measure_list,joinpar


# Temas pendientes
#  - heredar de dataframe 
#  - agregar variables de estado
#  - extender funcion resumen de dataframe
#  - modifficar elementos del objetos de manera segura
#  si df tiene varias columnas se requiere varx
# metodos posibles. remove nan (pero hereda dataframe?)
# los datos no se ordenan por defecto, pero si algún método lo requiere
# pasan a estar ordenados
class IneqMeasure:
    """
    Measuring Inequality, Poverty, Welfare, Polarization and Concentration.
    """    
    def __init__(self, df, varx=None, weight=None, sort=False):
        self.data = df
        self.varx = varx
        self.weight = weight
        self.sort = sort
        self.validate = self._validate()
    
    def _validate(self):
        if not isinstance(self.data, pd.DataFrame):            
            try:          
                self.data = pd.DataFrame(self.data)  
            except ValueError:
                print("No se puede convertir a un DataFrame")
        if self.varx==None:
            if self.weight==None:
                if self.data.shape[1]==1:            
                    self.varx = self.data.columns[0]
                else: 
                    raise ValueError("Se requiere el nombre de la variable (cols>1).")
            else:        
                raise ValueError("Se requiere el nombre de la variable.")
        elif not (self.varx in self.data.columns):
            raise ValueError("El nombre de la variable no existe.")
        if not(self.weight==None):          
            if not (self.weight in self.data.columns):
                raise ValueError("El nombre de la variable frecuencia no existe.")
        if not isinstance(self.sort, bool):
            raise ValueError("El argumento sort debe ser bool.")
        return True

    def _get_arg(self,sort):
        # ordena dataframe si es requerido
        if sort and (not self.sort):
            self.data = self.data.sort_values(by=self.varx)
            self.sort = True
        y = self.data[self.varx].values    
        if self.weight==None:
            w = None
            dow = False
        else:
            w = self.data[self.weight].values
            dow = True        
        return dow,y,w
            
    def poverty(self, method,*args): 
        dow,ys,w = self._get_arg(sort=True) 
        if dow:
            return poverty_measure_w(ys,w,method,*args)
        else:
            return poverty_measure(ys,method,*args)
        
    def tip(self,*args):    # Curve TIP
        dow,ys,w = self._get_arg(sort=True)
        if dow:
            raise ValueError("No implementado (datos agrupados).")
        else:
            return tip_curve(ys,*args)  
        
    def ineq(self, method,*args): 
        dow,ys,w = self._get_arg(sort=True) 
        if dow:
            return ineq_measure_w(ys,w,method,*args)
        else:
            return ineq_measure(ys,method,*args)

    def lorenz(self,*args):    # Curve de Lorenz
        dow,ys,w = self._get_arg(sort=True)
        if dow:
            raise ValueError("No implementado (datos agrupados).")
        else:
            return lorenz_curve(ys,*args)          
        
    def welfare(self, method,*args): 
        dow,ys,w = self._get_arg(sort=True) 
        if dow:
            return welfare_measure_w(ys,w,method,*args)
        else:
            return welfare_measure(ys,method,*args)   
        
    def polar(self, method,*args): 
        dow,ys,w = self._get_arg(sort=True) 
        if dow:
            return polarization_measure_w(ys,w,method,*args)
        else:
            return polarization_measure(ys,method,*args)   
        
    def conc(self, method,*args): 
        dow,ys,w = self._get_arg(sort=True) 
        if dow:
            return concentration_measure_w(ys,w,method,*args)
        else:
            return concentration_measure(ys,method,*args)   
        
        