# APODE Tutorial

Este documento muestra la funcionalidad actual del paquete apode. El mismoe dispone de varios métodos que calculan medidas y generan gráficos en los siguientes temas:

* Pobreza
* Desigualdad
* Bienestar
* Polarización
* Concentración

Otros temas serán agregados más adelante.

Al momento los algoritmos no han sido testeados y es escasa la documentación. 

## Table of Contents

- [Clase ApodeData](#clase-apodedata)
- [Data Creation and Description](#data-creation-and-description)
  * [Carga manual](#carga-manual)
  * [Lectura desde la web](#lectura-desde-la-web)
  * [Simulación](#simulaci-n)
- [Measures](#measures)
  * [Poverty](#poverty)
    + [Numerical measures](#numerical-measures)
    + [Graph measures](#graph-measures)
  * [Inequality](#inequality)
    + [Numerical measures](#numerical-measures-1)
    + [Graph measures](#graph-measures-1)
  * [Welfare](#welfare)
  * [Polarization](#polarization)
  * [Concentration](#concentration)
- [Tools](#tools)
  * [Decomposition](#decomposition)
  * [Comparative statics](#comparative-statics)
  * [Estimation](#estimation)
- [Todo](#todo)
- [Other packages](#other-packages)
- [References](#references)



# Clase ApodeData

Los objetos se crean mediante:

    df = ApodeData(DataFrame,varx)
    
En donde varx es el nombre de una columna del dataframe.

Metodos que calculan indicadores:
   
    df.poverty(method,*args)    
    df.ineq(method,*args)
    df.welfare(method,*args) 
    df.polar(method,*args)
    df.conc(method,*args)
 
Métodos que computan gráficos:

    df.tip(*args,**kwargs)
    df.lorenz(*args,**kwargs)
    df.pen(*args,**kwargs)
    
    

# Data Creation and Description

* Los datos pueden generarse manualmente o mediante simuación. Estan contenidos en un DataFrame
* Pueden existir otras variables categóricas que permiten aplicar los indicadores por grupos (groupby)


```python
import matplotlib.pyplot as plt
import seaborn as sns

import numpy as np
import pandas as pd
```


```python
import sys
sys.path.append("D:/GitHub/apode/")  # ver

from apode import ApodeData
from apode.distributions import distribution_examples #,default_rng # test  
```

## Carga manual

Se puede crear objeto desde un DataFrame o desde un argumento válido de la funcion DataFrame. La función *binning* se puede usar para agrupar datos.


```python
x = [23, 10, 12, 21, 4, 8, 19, 15, 11, 9]
df1 = pd.DataFrame({'x':x})
dr1 = ApodeData(df1, varx="x") 

dr1
```




    ApodeData(data=    x
    0  23
    1  10
    2  12
    3  21
    4   4
    5   8
    6  19
    7  15
    8  11
    9   9, varx='x', poverty=PovertyMeasures(idf=...), inequality=InequalityMeasures(idf=...), polarization=PolarizationMeasures(idf=...), concentration=ConcentrationMeasures(idf=...), welfare=WelfareMeasures(idf=...))



## Lectura desde la web

Usar LIS Database https://www.lisdatacenter.org/our-data/lis-database/

The Luxembourg Income Study Database (LIS) is the largest available income database of harmonised microdata collected from about 50 countries in Europe, North America, Latin America, Africa, Asia, and Australasia spanning five decades.


## Simulación 

La función *distribution_examples* brinda algunos ejemplos de distribuciones usuales para modelar la distribución del ingreso.


```python
# Generar datos
n = 1000 # observaciones
j_d = 6  # elegir distribción
rg = np.random.default_rng(12345)
listd = ['uniform','lognormal','exponential','pareto','chisquare','gamma','weibull']
fdistr = listd[j_d]
df2 = distribution_examples(rg,fdistr,n)

# Crear objeto (sin agrupamiento)
dr2 = ApodeData(df2,varx="x") 

# Graficar distribución
sns.distplot(df2).set_title(fdistr)
plt.show()
```


![png](output_8_0.png)


# Measures

## Poverty

Están implementados diversas medidas de pobreza y la curva TIP (permite comparar gráficamente la pobreza entre distribuciones).

Todos los métodos requieren la linea de pobreza (pline) y algunos métodos requieren un parámetro adicional (alpha). En algunos casos se aplica un valor por defecto.

### Numerical measures


```python
pline = 50 # Poverty line
p = dr2.poverty('headcount',pline=pline)
p
```




    0.656




```python
# Evaluar un listado de métodos
lista = [["headcount", None],
         ["gap", None],
         ["severity",None],
         ["fgt",1.5],
         ["sen",None],
         ["sst",None],
         ["watts",None],
         ["cuh",0],
         ["cuh",0.5],
         ["takayama",None],
         ["kakwani",None],
         ["thon",None],
         ["bd",1.0],
         ["bd",2.0],
         ["hagenaars",None],
         ["chakravarty",0.5]]
p = []
for elem in lista:
    if elem[1]==None:
        p.append(dr2.poverty(elem[0],pline=pline))
    else:
        p.append(dr2.poverty(elem[0],pline=pline,alpha=elem[1]))
        df_p = pd.concat([pd.DataFrame(lista),pd.DataFrame(p)],axis=1)
df_p.columns = ['method','alpha','measure']
df_p   
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>method</th>
      <th>alpha</th>
      <th>measure</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>headcount</td>
      <td>NaN</td>
      <td>0.656000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>gap</td>
      <td>NaN</td>
      <td>0.312483</td>
    </tr>
    <tr>
      <th>2</th>
      <td>severity</td>
      <td>NaN</td>
      <td>0.192512</td>
    </tr>
    <tr>
      <th>3</th>
      <td>fgt</td>
      <td>1.5</td>
      <td>0.192512</td>
    </tr>
    <tr>
      <th>4</th>
      <td>sen</td>
      <td>NaN</td>
      <td>0.437672</td>
    </tr>
    <tr>
      <th>5</th>
      <td>sst</td>
      <td>NaN</td>
      <td>0.279693</td>
    </tr>
    <tr>
      <th>6</th>
      <td>watts</td>
      <td>NaN</td>
      <td>0.549612</td>
    </tr>
    <tr>
      <th>7</th>
      <td>cuh</td>
      <td>0.0</td>
      <td>0.426800</td>
    </tr>
    <tr>
      <th>8</th>
      <td>cuh</td>
      <td>0.5</td>
      <td>0.359553</td>
    </tr>
    <tr>
      <th>9</th>
      <td>takayama</td>
      <td>NaN</td>
      <td>-0.302945</td>
    </tr>
    <tr>
      <th>10</th>
      <td>kakwani</td>
      <td>NaN</td>
      <td>0.458396</td>
    </tr>
    <tr>
      <th>11</th>
      <td>thon</td>
      <td>NaN</td>
      <td>0.484403</td>
    </tr>
    <tr>
      <th>12</th>
      <td>bd</td>
      <td>1.0</td>
      <td>0.394687</td>
    </tr>
    <tr>
      <th>13</th>
      <td>bd</td>
      <td>2.0</td>
      <td>0.507942</td>
    </tr>
    <tr>
      <th>14</th>
      <td>hagenaars</td>
      <td>NaN</td>
      <td>0.140493</td>
    </tr>
    <tr>
      <th>15</th>
      <td>chakravarty</td>
      <td>0.5</td>
      <td>0.199721</td>
    </tr>
  </tbody>
</table>
</div>



### Graph measures


```python
# Curva TIP
# dr2.poverty("tip",pline=pline,plot=True)
df_tip = dr2.poverty("tip",pline=pline)
```


![png](output_15_0.png)


## Inequality

Están implementadas 12 medidas de desigualdad y la Curva de Lorenz relativa, gemeralizada y absoluta (permite comparar gráficamente la desigualdad entre distribuciones)

### Numerical measures


```python
# Evaluar un método 
q = dr2.inequality('gini')
q
```




    0.3644326926296162




```python
# Evaluar un listado de métodos
lista = [["rrange", None],
         ["rad", None],
         ["cv",None],
         ["sdlog",None],
         ["gini",None],
         ["merhan",None],
         ["piesch",None],
         ["bonferroni",None],
         ["kolm",0.5],
         ["ratio",0.05],
         ["ratio",0.2],
         ["entropy",0],
         ["entropy",1],
         ["entropy",2],         
         ["atkinson",0.5],
         ["atkinson",1.0],
         ["atkinson",2.0]]
p = []
for elem in lista:
    if elem[1]==None:
        p.append(dr2.inequality(elem[0]))
    else:
        p.append(dr2.inequality(elem[0],alpha=elem[1]))

df_i = pd.concat([pd.DataFrame(lista),pd.DataFrame(p)],axis=1)
df_i.columns = ['method','alpha','measure']
df_i   
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>method</th>
      <th>alpha</th>
      <th>measure</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>rrange</td>
      <td>NaN</td>
      <td>4.154631</td>
    </tr>
    <tr>
      <th>1</th>
      <td>rad</td>
      <td>NaN</td>
      <td>0.264170</td>
    </tr>
    <tr>
      <th>2</th>
      <td>cv</td>
      <td>NaN</td>
      <td>0.666949</td>
    </tr>
    <tr>
      <th>3</th>
      <td>sdlog</td>
      <td>NaN</td>
      <td>0.887906</td>
    </tr>
    <tr>
      <th>4</th>
      <td>gini</td>
      <td>NaN</td>
      <td>0.364433</td>
    </tr>
    <tr>
      <th>5</th>
      <td>merhan</td>
      <td>NaN</td>
      <td>0.509579</td>
    </tr>
    <tr>
      <th>6</th>
      <td>piesch</td>
      <td>NaN</td>
      <td>0.288869</td>
    </tr>
    <tr>
      <th>7</th>
      <td>bonferroni</td>
      <td>NaN</td>
      <td>0.507513</td>
    </tr>
    <tr>
      <th>8</th>
      <td>kolm</td>
      <td>0.50</td>
      <td>34.194790</td>
    </tr>
    <tr>
      <th>9</th>
      <td>ratio</td>
      <td>0.05</td>
      <td>0.034212</td>
    </tr>
    <tr>
      <th>10</th>
      <td>ratio</td>
      <td>0.20</td>
      <td>0.121993</td>
    </tr>
    <tr>
      <th>11</th>
      <td>entropy</td>
      <td>0.00</td>
      <td>0.273518</td>
    </tr>
    <tr>
      <th>12</th>
      <td>entropy</td>
      <td>1.00</td>
      <td>0.216293</td>
    </tr>
    <tr>
      <th>13</th>
      <td>entropy</td>
      <td>2.00</td>
      <td>0.222410</td>
    </tr>
    <tr>
      <th>14</th>
      <td>atkinson</td>
      <td>0.50</td>
      <td>0.112985</td>
    </tr>
    <tr>
      <th>15</th>
      <td>atkinson</td>
      <td>1.00</td>
      <td>0.239301</td>
    </tr>
    <tr>
      <th>16</th>
      <td>atkinson</td>
      <td>2.00</td>
      <td>0.568995</td>
    </tr>
  </tbody>
</table>
</div>



### Graph measures

Un argumento es plot (valor true por defecto).


```python
# Curva de Lorenz
df_lor = dr2.inequality('lorenz')
```


![png](output_21_0.png)



```python
# Curva de Lorenz Generalizada
df_lorg = dr2.inequality('lorenz',alpha='g')
```


![png](output_22_0.png)



```python
# Curva de Lorenz Absoluta
df_lora = dr2.inequality('lorenz',alpha='a')
```


![png](output_23_0.png)



```python
# Pen's Parade
df_pen = dr2.inequality('pen',pline=60)
```


![png](output_24_0.png)


## Welfare

Están implementadas 5 funciones de bienestar social.


```python
# Evaluar un método 
w = dr2.welfare('sen')
w
```




    43.83069516538674




```python
# Evaluar un listado de métodos
lista = [["utilitarian", None],
         ["rawlsian", None],
         ["sen",None],
         ["theill",None],
         ["theilt",None],
         ["isoelastic",0],
         ["isoelastic",1],
         ["isoelastic",2],
         ["isoelastic",np.Inf]]
p = []
for elem in lista:
    if elem[1]==None:
        p.append(dr2.welfare(elem[0]))
    else:
        p.append(dr2.welfare(elem[0],alpha=elem[1]))

df_w = pd.concat([pd.DataFrame(lista),pd.DataFrame(p)],axis=1)
df_w.columns = ['method','alpha','measure']
df_w   
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>method</th>
      <th>alpha</th>
      <th>measure</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>utilitarian</td>
      <td>NaN</td>
      <td>43.165526</td>
    </tr>
    <tr>
      <th>1</th>
      <td>rawlsian</td>
      <td>NaN</td>
      <td>0.308786</td>
    </tr>
    <tr>
      <th>2</th>
      <td>sen</td>
      <td>NaN</td>
      <td>43.830695</td>
    </tr>
    <tr>
      <th>3</th>
      <td>theill</td>
      <td>NaN</td>
      <td>32.835971</td>
    </tr>
    <tr>
      <th>4</th>
      <td>theilt</td>
      <td>NaN</td>
      <td>34.769813</td>
    </tr>
    <tr>
      <th>5</th>
      <td>isoelastic</td>
      <td>0.0</td>
      <td>43.165526</td>
    </tr>
    <tr>
      <th>6</th>
      <td>isoelastic</td>
      <td>1.0</td>
      <td>3.491525</td>
    </tr>
    <tr>
      <th>7</th>
      <td>isoelastic</td>
      <td>2.0</td>
      <td>-0.053750</td>
    </tr>
    <tr>
      <th>8</th>
      <td>isoelastic</td>
      <td>inf</td>
      <td>0.308786</td>
    </tr>
  </tbody>
</table>
</div>



## Polarization 

Están implementados 2 medidas de polarización.


```python
# Evaluar un método 
p = dr2.polarization('ray')
p
```




    0.03146185773732036




```python
# Evaluar un listado de métodos
lista = [["ray", None],
         ["wolfson", None]]
p = []
for elem in lista:
    if elem[1]==None:
        p.append(dr2.polarization(elem[0]))
    else:
        p.append(dr2.polarization(elem[0],alpha=elem[1]))
df_pz = pd.concat([pd.DataFrame(lista),pd.DataFrame(p)],axis=1)
df_pz.columns = ['method','alpha','measure']
df_pz 
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>method</th>
      <th>alpha</th>
      <th>measure</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>ray</td>
      <td>None</td>
      <td>0.031462</td>
    </tr>
    <tr>
      <th>1</th>
      <td>wolfson</td>
      <td>None</td>
      <td>0.351678</td>
    </tr>
  </tbody>
</table>
</div>



## Concentration

Están implementadas 4 medidas de concentración (de uso comun para analizar la concentración industrial).


```python
# Evaluar un método
c = dr2.concentration('herfindahl')
c
```




    0.00044526570696696563




```python
# Evaluar un listado de métodos
lista = [["herfindahl", None],
         ["herfindahl", True],
         ["rosenbluth",None],
         ["concentration_ratio",1],
         ["concentration_ratio",5]]
p = []
for elem in lista:
    if elem[1]==None:
        p.append(dr2.concentration(elem[0]))
    else:
        if elem[0]=="herfindahl":
            p.append(dr2.concentration(elem[0],normalized=elem[1]))  # ver keyword
        elif elem[0]=="concentration_ratio":
            p.append(dr2.concentration(elem[0],k=elem[1]))  # ver keyword            
        else:
            p.append(dr2.concentration(elem[0],alpha=elem[1]))

df_c = pd.concat([pd.DataFrame(lista),pd.DataFrame(p)],axis=1)
df_c.columns = ['method','alpha','measure']
df_c 
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>method</th>
      <th>alpha</th>
      <th>measure</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>herfindahl</td>
      <td>None</td>
      <td>0.000445</td>
    </tr>
    <tr>
      <th>1</th>
      <td>herfindahl</td>
      <td>True</td>
      <td>0.000445</td>
    </tr>
    <tr>
      <th>2</th>
      <td>rosenbluth</td>
      <td>None</td>
      <td>0.001573</td>
    </tr>
    <tr>
      <th>3</th>
      <td>concentration_ratio</td>
      <td>1</td>
      <td>0.004162</td>
    </tr>
    <tr>
      <th>4</th>
      <td>concentration_ratio</td>
      <td>5</td>
      <td>0.017772</td>
    </tr>
  </tbody>
</table>
</div>



# Tools

## Decomposition

Los medidas pueden aplicarse por subrgrupos de acuerdo a cierta categoría. Por ejemplo:


```python
x = [23, 10, 12, 21, 4, 8, 19, 15, 5, 7]
y = [10,10,20,10,10,20,20,20,10,10] 
w = np.arange(1,11)
dfa = pd.DataFrame({'x':x,'y':y,'w':w})
dfa
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>x</th>
      <th>y</th>
      <th>w</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>23</td>
      <td>10</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>10</td>
      <td>10</td>
      <td>2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>12</td>
      <td>20</td>
      <td>3</td>
    </tr>
    <tr>
      <th>3</th>
      <td>21</td>
      <td>10</td>
      <td>4</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>10</td>
      <td>5</td>
    </tr>
    <tr>
      <th>5</th>
      <td>8</td>
      <td>20</td>
      <td>6</td>
    </tr>
    <tr>
      <th>6</th>
      <td>19</td>
      <td>20</td>
      <td>7</td>
    </tr>
    <tr>
      <th>7</th>
      <td>15</td>
      <td>20</td>
      <td>8</td>
    </tr>
    <tr>
      <th>8</th>
      <td>5</td>
      <td>10</td>
      <td>9</td>
    </tr>
    <tr>
      <th>9</th>
      <td>7</td>
      <td>10</td>
      <td>10</td>
    </tr>
  </tbody>
</table>
</div>




```python
# calculo simple
pline = 11
dra1 = ApodeData(dfa,varx='x') 
p1 = dra1.poverty('headcount',pline=pline)
p1
```




    0.5




```python
# ver si vale la pena implementarlo en la clase
# recibe un dataframe y aplica medida de acuerdo a columna "varg"
def poverty_gby(dfa,method,varg,pline):
    grouped = dfa.groupby(varg)
    a = []
    b = []
    c = []
    varx = 'x'
    for name, group in grouped:
        y = group[varx].values
        count = group.shape[0]
        dri = ApodeData({varx:y},varx=varx)
        p = dri.poverty(method,pline=pline)
        a.append(name)
        b.append(p)
        c.append(count)
    xname = varx + "_measure"
    wname =varx + "_weight"
    return pd.DataFrame({xname: b, wname: c}, index=pd.Index(a))    
```


```python
# calculo por grupos según variable "y"
p2 = poverty_gby(dfa,'headcount',varg='y',pline=pline)
p2
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>x_measure</th>
      <th>x_weight</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>10</th>
      <td>0.666667</td>
      <td>6</td>
    </tr>
    <tr>
      <th>20</th>
      <td>0.250000</td>
      <td>4</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Si el indicador es descomponible se obtiene el mismo resultado:
p2_p = sum(p2['x_measure']*p2['x_weight']/sum(p2['x_weight']))
p2_p
```




    0.5



## Comparative statics

Comparación de las medidas  en dos monetos de tiempo. 

Matrices de transición.

## Estimation

Se puede estimar:

* Intervalos de confianza de los indicadores usando bootstrap 
* Distribuciones paramétricas de los datos (algunas permiten calcular indirectamente las medidas, Pareto y Gini, por ej)

# Todo

**En algoritmos falta (implementación):**

* Tamaño nulo del dataframe
* Division por cero (/log(1))
* overflow
* Tratamiento de missings
* implementación eficiente (algunos son lentos: polarizacion)
* mejorar algunos nombres
* ver si uniformar nombre de parametros
* Hay metodos que tienen varios nombres o que pueden estar en diferentes categorías. Ver si agregar redundancia.


**En test:**

* Comparar resultados con librerias de R (y Stata)

# Other packages

Paquetes relacionados.

**Python**

- http://www.poorcity.richcity.org/oei/  (algoritmos)
- https://github.com/mmngreco/IneqPy
- https://pythonhosted.org/IneqPy/ineqpy.html
- https://github.com/open-risk/concentrationMetrics
- https://github.com/cjohnst5/GBdistributiontree

**R**

- https://cran.r-project.org/web/packages/ineq/ineq.pdf
- https://cran.r-project.org/web/packages/affluenceIndex/affluenceIndex.pdf
- https://cran.r-project.org/web/packages/dineq/dineq.pdf
- https://github.com/PABalland/EconGeo
- https://cran.r-project.org/web/packages/rtip/rtip.pdf
- https://cran.r-project.org/web/packages/GB2/index.html

**Stata**

- https://www.stata.com/manuals/rinequality.pdf
- http://dasp.ecn.ulaval.ca/dmodules/madds20.htm


# References

* Cowell, F. (2011) Measuring Inequality. London School of Economics Perspectives in Economic Analysis. 3rd ed. Edición. Oxford University Press
http://darp.lse.ac.uk/papersDB/Cowell_measuringinequality3.pdf
* Cowell, F. (2016) “Inequality and Poverty Measures”, in Oxford Handbook of Well-Being And Public Policy, edited by Matthew D. Adler and Marc Fleurbaey 
* Haughton, J. and S. Khandker (2009). Handbook on Poverty + Inequality. World Bank Training Series. https://openknowledge.worldbank.org/bitstream/handle/10986/11985/9780821376133.pdf
* POBREZA Y DESIGUALDAD EN AMÉRICA LATINA. https://www.cedlas.econo.unlp.edu.ar/wp/wp-content/uploads/Pobreza_desigualdad_-America_Latina.pdf
* https://www.cepal.org/es/publicaciones/4740-enfoques-la-medicion-la-pobreza-breve-revision-la-literatura
* https://www.cepal.org/es/publicaciones/4788-consideraciones-indice-gini-medir-la-concentracion-ingreso



