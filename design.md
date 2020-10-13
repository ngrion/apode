# Estructura del paquete apode

## Objetivos

* Aplicar diferentes medidas a un vector de datos, con estas variantes:
    - Los datos pueden estar agrupados ([frecuencia, valor])
    - Las medidas pueden aplicarse a subconjuntos de los datos  (descomposición de acuerdo a alguna variable categorica)

## Estructura

Actualmente el paquete está formado por tres partes independientes:

* Generación de datos
* Clase IneqMeasure
* Algoritmos

### Generación de datos

Se recurre a carga manual, fuentes externas o generadores (numpy.random). No se usa la clase (los datos pueden estar en una lista, dataframe, etc.). 

### Clase IneqMeasure

La clase está formada por:

* un dataframe que contiene la/s variable/s. Si se crea el objeto y no se le pasa un dataframe, intenta convertirlo (se puede pasar una lista)
* parametros (opcionales):
    - Si hay más de una columna un parámetro indica el nombre de la variable principal. 
    - Lo mismo sucede si existe otra columa con ponderadores. 
    - Por eficiencia otro parámetro permite indicar si los datos ya están ordenados.
* Existen cuatro tipos de métodos:
    1) Para validar todos los argumentos (automático al crear el objeto). 
    2) Para acceder y modificar los atributos
    3) Para generar los argumentos para todos los métodos de 4)
    4) Indicadores. Todos tienen la misma estructura. Se pueden aplicar al total de datos o por subgrupos.

Sobre el punto 4), por ejemplo, se tiene:

    def poverty(self, method,*args) 
    def ineq(self, method,*args)

Se podría reemplazar todos los tipos de métodos por un solo método:

    def measure(self, method,*args)     

El nombre del método tiene implícito el tipo de medida (pobreza, desigualdad, etc). Esto simplifica el código pero creo que lo hace menos intuitivo.

### Indicadores

Es la parte que más programación insume porque se implementan decenas de métodos. Pero todos tienen la misma interface:
    
    escalar   = f(variable, ponderador,par1,par2,...) 
    dataframe = f(variable, ponderador,par1,par2,...,gby=var)  # descomposicion

Por ejemplo, para pobreza (se diferencia si son datos agrupados o no):

    p = poverty_measure_w(ys,w,method,*args)
    p = poverty_measure(ys,method,*args)

En la codificación no se usa la clase. Esto permite mejorar los algoritmos o agregar nuevos sin modificar la clase.

### Tareas Pendientes

* Test 
    - Realizar test usando resultados de las librerías de R.
    - Validar parametros válidos en los indicadores
* Crear una clase de error

