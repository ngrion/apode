#!/usr/bin/env python
# -*- coding: utf-8 -*-

# This file is part of the
#   Apode Project (https://github.com/mchalela/apode).
# Copyright (c) 2020, Néstor Grión and Sofía Sappia
# License: MIT
#   Full Text: https://github.com/ngrion/apode/blob/master/LICENSE.txt


import pytest
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from apode.basic import ApodeData
from apode.poverty import PovertyMeasures 

from numpy.testing import assert_equal, assert_, assert_almost_equal
from numpy import random

class TestCaseUniform:
    def setup_method(self):
        x = np.random.uniform(size=10)
        df1 = pd.DataFrame({'x':x})
        self.data = ApodeData(df1, varx="x") 

    def test_min(self,method,**kwargs):
        pline = np.min(self.data.data.values)-1
        assert self.data.poverty(method,pline=pline,**kwargs) == 0

    def test_max(self,method,**kwargs):
        pline = np.max(self.data.data.values)+1
        assert self.data.poverty(method,pline=pline,**kwargs) == 1        
        
    def test_symmetry(self,method,**kwargs):
        pline = np.mean(self.data.data.values)
        y = self.data.data['x'].tolist()
        random.shuffle(y)        
        df2 = pd.DataFrame({'x':y})
        dr2 = ApodeData(df2, varx="x")
        assert self.data.poverty(method,pline=pline,**kwargs) == dr2.poverty(method,pline=pline,**kwargs)
        
    def test_replication(self,method,**kwargs):
        k = 2 # factor
        pline = np.mean(self.data.data.values)
        y = k*self.data.data['x'].tolist()     
        df2 = pd.DataFrame({'x':y})
        dr2 = ApodeData(df2, varx="x")
        assert self.data.poverty(method,pline=pline,**kwargs) == dr2.poverty(method,pline=pline,**kwargs)        
                
    def test_homogeneity(self,method,**kwargs):
        k = 2 # factor
        pline = np.mean(self.data.data.values)
        y = self.data.data['x'].tolist()   
        y = [yi * k for yi in y]
        df2 = pd.DataFrame({'x':y})
        dr2 = ApodeData(df2, varx="x")
        assert self.data.poverty(method,pline=pline,**kwargs) == dr2.poverty(method,pline=pline*k,**kwargs)    


# Testea metdodo de un listado de medidas de pobreza
def test_lista(prop,lista):
    x = TestCaseUniform()
    x.setup_method()
    for elem in lista:
        if elem[1]==None:
            x.prop(elem[0])
        else:
            x.prop(elem[0],alpha=elem[1])        