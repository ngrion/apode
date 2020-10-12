# APODE

Este documento muestra la funcionalidad actual del paquete apode. El mismo provee una clase que hereda (pendiente) de la clase DataFrame.

Esta clase dispone de varios métodos que calculan medidas y generan gráficos en los siguuinetes temas:

* Pobreza
* Desigualdad
* Bienestar
* Polarización
* Concentración

Otros temas serán agregados más adelante.

Al momento los algoritmos no han sido testeados y es escasa la documentación. 



```python
import matplotlib.pyplot as plt
import seaborn as sns

import numpy as np
import pandas as pd

from apode import IneqMeasure # clase
from apode import distribution_examples,joinpar # test
```

# Generar datos

* Los datos pueden generarse manualmente o mediante simuación. Estan contenidos en un DataFrame
* Los datos pueden estar agrupados. En este caso una variable contiene las frecuencias.
* Pueden existir otras variables categóricas que permiten aplicar los indicadores por grupos (groupby)
* Un parámetros indica si los datos están ordenados (por defecto no)


## Carga manual

Se puede crear objeto desde un DataFrame o desde un argumento válido de la funcion DataFrame. La función *binning* se puede usar para agrupar datos.


```python
# dr1a y dr1b son equivalentes
x = [23, 10, 12, 21, 4, 8, 19, 15, 11, 9]
dr1a = IneqMeasure(x) 

df1 = pd.DataFrame({'x':x})
dr1b = IneqMeasure(df1) 
```

## Lectura desde la web

Falta de implementar. 

## Simulación 

La función *distribution_examples* brinda algunos ejemplos de distribuciones usuales para modelar la distribución del ingreso.

Se generan dos objetos (datos agrupados y no agrupados) que serán utilizados más adelante para mostrar la aplicación de diferentes medidas (muchos de las medidas aún no estan implementadas para datos agrupados).

### Datos no agregados


```python
# Generar datos
n = 1000 # observaciones
j_d = 6  # elegir distribción
listd = ['uniform','lognormal','exponential','pareto','chisquare','gamma','weibull']
fdistr = listd[j_d]
df2 = distribution_examples(fdistr,n)

# Crear objeto (sin agrupamiento)
dr2 = IneqMeasure(df2) 

# Graficar distribución
sns.distplot(df2).set_title(fdistr)
plt.show()
```


![png](output_6_0.png)


### Datos agregados


```python
# Generar datos con agrupamiento
nbins = 10 # maximo, se descartan NAN
df3 = distribution_examples(fdistr,n,nbins)

# Crear objeto
dr3 = IneqMeasure(df3,varx = 'x',weight='weight')  

df3
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
      <th>weight</th>
      <th>x</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>227</td>
      <td>22.779185</td>
    </tr>
    <tr>
      <th>1</th>
      <td>293</td>
      <td>60.124881</td>
    </tr>
    <tr>
      <th>2</th>
      <td>212</td>
      <td>97.720530</td>
    </tr>
    <tr>
      <th>3</th>
      <td>135</td>
      <td>136.913947</td>
    </tr>
    <tr>
      <th>4</th>
      <td>63</td>
      <td>175.519930</td>
    </tr>
    <tr>
      <th>5</th>
      <td>41</td>
      <td>215.155125</td>
    </tr>
    <tr>
      <th>6</th>
      <td>16</td>
      <td>248.632630</td>
    </tr>
    <tr>
      <th>7</th>
      <td>11</td>
      <td>291.817267</td>
    </tr>
    <tr>
      <th>8</th>
      <td>1</td>
      <td>336.448536</td>
    </tr>
    <tr>
      <th>9</th>
      <td>1</td>
      <td>393.714140</td>
    </tr>
  </tbody>
</table>
</div>



# Poverty

EStán implementados 11 medidas de pobreza y la curva TIP (permite comparar gráficamente la pobreza entre distribuciones)


```python
pline = 50 # Poverty line
# Evaluar un método - datos sin agrupar
p = dr2.poverty('fgt0',pline)
p
```




    0.28




```python
# Evaluar un método - datos agrupados
p = dr3.poverty('fgt0',pline)
p
```




    0.227




```python
# Evaluar un listado de métodos
mlist_p = ['fgt0','fgt1','fgt2',['fgt',1.5],'sen','sst','watts',['cuh',0],['cuh',0.5],'takayama','kakwani','thon',['bd',2],'hagenaars',['chakravarty',0.5 ]]
mlist_p2 = [joinpar(x,pline) for x in mlist_p ]
table = []
for elem in mlist_p2:   
    table.append(dr2.poverty(elem[0],*elem[1:]))
df_outp =  pd.DataFrame(mlist_p2,columns = ['method','pline','par'])  
df_outp['poverty_measure'] = table
df_outp
```

    D:\EquipoDrive\Cursos Tomados\Diseño de software - FAMAF\apode2020\apode\poverty.py:76: RuntimeWarning: overflow encountered in int_scalars
      p = (q/(n*pline*a))*u
    




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
      <th>pline</th>
      <th>par</th>
      <th>poverty_measure</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>fgt0</td>
      <td>50</td>
      <td>NaN</td>
      <td>0.280000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>fgt1</td>
      <td>50</td>
      <td>NaN</td>
      <td>0.123963</td>
    </tr>
    <tr>
      <th>2</th>
      <td>fgt2</td>
      <td>50</td>
      <td>NaN</td>
      <td>0.074435</td>
    </tr>
    <tr>
      <th>3</th>
      <td>fgt</td>
      <td>50</td>
      <td>1.5</td>
      <td>0.093826</td>
    </tr>
    <tr>
      <th>4</th>
      <td>sen</td>
      <td>50</td>
      <td>NaN</td>
      <td>0.165390</td>
    </tr>
    <tr>
      <th>5</th>
      <td>sst</td>
      <td>50</td>
      <td>NaN</td>
      <td>0.043925</td>
    </tr>
    <tr>
      <th>6</th>
      <td>watts</td>
      <td>50</td>
      <td>NaN</td>
      <td>0.215153</td>
    </tr>
    <tr>
      <th>7</th>
      <td>cuh</td>
      <td>50</td>
      <td>0.0</td>
      <td>0.199133</td>
    </tr>
    <tr>
      <th>8</th>
      <td>cuh</td>
      <td>50</td>
      <td>0.5</td>
      <td>0.151399</td>
    </tr>
    <tr>
      <th>9</th>
      <td>takayama</td>
      <td>50</td>
      <td>NaN</td>
      <td>0.113481</td>
    </tr>
    <tr>
      <th>10</th>
      <td>kakwani</td>
      <td>50</td>
      <td>NaN</td>
      <td>33.864708</td>
    </tr>
    <tr>
      <th>11</th>
      <td>thon</td>
      <td>50</td>
      <td>NaN</td>
      <td>0.225275</td>
    </tr>
    <tr>
      <th>12</th>
      <td>bd</td>
      <td>50</td>
      <td>2.0</td>
      <td>-694.071806</td>
    </tr>
    <tr>
      <th>13</th>
      <td>hagenaars</td>
      <td>50</td>
      <td>NaN</td>
      <td>0.054998</td>
    </tr>
    <tr>
      <th>14</th>
      <td>chakravarty</td>
      <td>50</td>
      <td>0.5</td>
      <td>0.078805</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Curva TIP
df_tip = dr2.tip(pline)
```


![png](output_13_0.png)


# Inequality

Están implementadas 12 medidas de desigualdad y la Curva de Lorenz (permite comparar gráficamente la desigualdad entre distribuciones)


```python
# Evaluar un método - datos sin agrupar
q = dr2.ineq('gini')
q
```




    0.37336840083953776




```python
# Evaluar un método - datos agrupados
q = dr3.ineq('rr')
q
```




    4.13136931137181




```python
# Evaluar una lista de métodos
list_i = ['rr','dmr','cv','dslog','gini','merhan','piesch','bonferroni',['kolm',0.5],['ratio',0.05],['ratio',0.2], \
          ['entropy',0],['entropy',1],['entropy',2],['atkinson',0.5],['atkinson',1],['atkinson',2]]
list_i = [[elem] if not isinstance(elem,list) else elem for elem in list_i]
table = []
for elem in list_i:   
    table.append(dr2.ineq(*elem))
dz_i =  pd.DataFrame(list_i,columns = ['method','par'])  
dz_i['ineq_measure'] = table
dz_i
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
      <th>par</th>
      <th>ineq_measure</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>rr</td>
      <td>NaN</td>
      <td>4.190042</td>
    </tr>
    <tr>
      <th>1</th>
      <td>dmr</td>
      <td>NaN</td>
      <td>0.267542</td>
    </tr>
    <tr>
      <th>2</th>
      <td>cv</td>
      <td>NaN</td>
      <td>0.699532</td>
    </tr>
    <tr>
      <th>3</th>
      <td>dslog</td>
      <td>NaN</td>
      <td>0.920803</td>
    </tr>
    <tr>
      <th>4</th>
      <td>gini</td>
      <td>NaN</td>
      <td>0.373368</td>
    </tr>
    <tr>
      <th>5</th>
      <td>merhan</td>
      <td>NaN</td>
      <td>0.521567</td>
    </tr>
    <tr>
      <th>6</th>
      <td>piesch</td>
      <td>NaN</td>
      <td>0.299272</td>
    </tr>
    <tr>
      <th>7</th>
      <td>bonferroni</td>
      <td>NaN</td>
      <td>0.519523</td>
    </tr>
    <tr>
      <th>8</th>
      <td>kolm</td>
      <td>0.50</td>
      <td>82.761199</td>
    </tr>
    <tr>
      <th>9</th>
      <td>ratio</td>
      <td>0.05</td>
      <td>0.028309</td>
    </tr>
    <tr>
      <th>10</th>
      <td>ratio</td>
      <td>0.20</td>
      <td>0.109784</td>
    </tr>
    <tr>
      <th>11</th>
      <td>entropy</td>
      <td>0.00</td>
      <td>0.293319</td>
    </tr>
    <tr>
      <th>12</th>
      <td>entropy</td>
      <td>1.00</td>
      <td>0.232264</td>
    </tr>
    <tr>
      <th>13</th>
      <td>entropy</td>
      <td>2.00</td>
      <td>0.244672</td>
    </tr>
    <tr>
      <th>14</th>
      <td>atkinson</td>
      <td>0.50</td>
      <td>-82384.287203</td>
    </tr>
    <tr>
      <th>15</th>
      <td>atkinson</td>
      <td>1.00</td>
      <td>-69875.279931</td>
    </tr>
    <tr>
      <th>16</th>
      <td>atkinson</td>
      <td>2.00</td>
      <td>-40019.793047</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Curva de Lorenz
df_lor = dr2.lorenz()
```


![png](output_18_0.png)


# Welfare

Están implementadas 5 funciones de bienestar social.


```python
# Evaluar un método - datos sin agrupar
w = dr2.welfare('sen')
w
```




    58.71267654908107




```python
# Evaluar un método - datos agrupados
w = dr3.welfare('utilitarian')
w
```




    89.78499069611702




```python
# Lista de indicadores
list_w = ['utilitarian','rawlsian','sen','theill','theilt',['isoelastic',0],['isoelastic',1],['isoelastic',2],['isoelastic',np.Inf]]
list_w = [[elem] if not isinstance(elem,list) else elem for elem in list_w]
table = []
for elem in list_w:   
    table.append(dr2.welfare(*elem))
dz_w =  pd.DataFrame(list_w,columns = ['method','par'])  
dz_w['welfare_measure'] = table
dz_w     
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
      <th>par</th>
      <th>welfare_measure</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>utilitarian</td>
      <td>NaN</td>
      <td>93.695086</td>
    </tr>
    <tr>
      <th>1</th>
      <td>rawlsian</td>
      <td>NaN</td>
      <td>1.112989</td>
    </tr>
    <tr>
      <th>2</th>
      <td>sen</td>
      <td>NaN</td>
      <td>58.712677</td>
    </tr>
    <tr>
      <th>3</th>
      <td>theill</td>
      <td>NaN</td>
      <td>69.876280</td>
    </tr>
    <tr>
      <th>4</th>
      <td>theilt</td>
      <td>NaN</td>
      <td>74.275522</td>
    </tr>
    <tr>
      <th>5</th>
      <td>isoelastic</td>
      <td>0.0</td>
      <td>93.695086</td>
    </tr>
    <tr>
      <th>6</th>
      <td>isoelastic</td>
      <td>1.0</td>
      <td>4.246726</td>
    </tr>
    <tr>
      <th>7</th>
      <td>isoelastic</td>
      <td>2.0</td>
      <td>-0.024987</td>
    </tr>
    <tr>
      <th>8</th>
      <td>isoelastic</td>
      <td>inf</td>
      <td>1.112989</td>
    </tr>
  </tbody>
</table>
</div>



# Polarization 

Están implementados 2 medidas de polarización.


```python
# Evaluar un método - datos sin agrupar
p = dr2.polar('er')
p
```




    0.0703395998697128




```python
# lista de indicadores
list_pz = ['er','wlf']
table = []
for elem in list_pz:   
    table.append(dr2.polar(elem))
dz_pz =  pd.DataFrame(list_pz,columns = ['method'])  
dz_pz['polarization_measure'] = table
dz_pz
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
      <th>polarization_measure</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>er</td>
      <td>0.070340</td>
    </tr>
    <tr>
      <th>1</th>
      <td>wlf</td>
      <td>-0.126431</td>
    </tr>
  </tbody>
</table>
</div>



# Concentration

Están implementadas 4 medidas de concentración (de uso comun para analizar la concentración industrial).


```python
# Evaluar un método - datos sin agrupar
c = dr2.conc('hhi')
c
```




    0.001489344687416868




```python
# lista de indicadores
list_c = ['hhi','hhin','rosenbluth',['cr',1],['cr',5]]
list_c = [[elem] if not isinstance(elem,list) else elem for elem in list_c]
table = []
for elem in list_c:   
    table.append(dr2.conc(*elem))
dz_c =  pd.DataFrame(list_c,columns = ['method','par'])  
dz_c['concentration_measure'] = table
dz_c     
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
      <th>par</th>
      <th>concentration_measure</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>hhi</td>
      <td>NaN</td>
      <td>0.001489</td>
    </tr>
    <tr>
      <th>1</th>
      <td>hhin</td>
      <td>NaN</td>
      <td>0.000490</td>
    </tr>
    <tr>
      <th>2</th>
      <td>rosenbluth</td>
      <td>NaN</td>
      <td>0.001596</td>
    </tr>
    <tr>
      <th>3</th>
      <td>cr</td>
      <td>1.0</td>
      <td>0.004202</td>
    </tr>
    <tr>
      <th>4</th>
      <td>cr</td>
      <td>5.0</td>
      <td>0.019779</td>
    </tr>
  </tbody>
</table>
</div>



# Pendiente


**En algoritmos falta (implementación):**

* valores no válidos (x y w, parámetros)
* Tamaño nulo del dataframe
* Division por cero (/log(1))
* overflow
* Tratamiento de missings
* implementación eficiente (algunos son lentos: polarizacion)
* mejorar algunos nombres
* Hay metodos que tienen varios nombres o que pueden estar en diferentes categorías. Ver si agregar redundancia.
* Curva de lorenz generalizada
* random (cambiar generador)


**En algoritmos (análisis):**

* Intervalos de confianza
* comparación de distribuciones (impacto de políticas, etc)
* descomposición (basarse en groupby)


**En test:**

* Comparar resultados con librerias de R

# Otras Implementaciones

Relevantes para el dearrollo de los test.

**Python**

- http://www.poorcity.richcity.org/oei/  (algoritmos)
- https://github.com/mmngreco/IneqPy
- https://pythonhosted.org/IneqPy/ineqpy.html
- https://github.com/open-risk/concentrationMetrics

**R**

- https://cran.r-project.org/web/packages/ineq/ineq.pdf
- https://cran.r-project.org/web/packages/affluenceIndex/affluenceIndex.pdf
- https://cran.r-project.org/web/packages/dineq/dineq.pdf
- https://github.com/PABalland/EconGeo
- https://cran.r-project.org/web/packages/rtip/rtip.pdf

**Stata**

- https://www.stata.com/manuals/rinequality.pdf
- http://dasp.ecn.ulaval.ca/dmodules/madds20.htm


# Referencias

* F A Cowell: Measuring Inequality, 1995 Prentice Hall
* Handbook on Poverty and Inequality. https://openknowledge.worldbank.org/bitstream/handle/10986/11985/9780821376133.pdf
* POBREZA Y DESIGUALDAD EN AMÉRICA LATINA. https://www.cedlas.econo.unlp.edu.ar/wp/wp-content/uploads/Pobreza_desigualdad_-America_Latina.pdf
* https://www.cepal.org/es/publicaciones/4740-enfoques-la-medicion-la-pobreza-breve-revision-la-literatura
* https://www.cepal.org/es/publicaciones/4788-consideraciones-indice-gini-medir-la-concentracion-ingreso



