# Notas sobre el paquete


# Métodos



## Indicadores 

Los principales son de pobreza y desigualdad. Algunas “medidas” no tienen como salida un escalar sino un gráfico (Lorenz). Pongo un ejemplo de cada uno (material más técnico se lista en el apéndice):



*   Pobreza
[https://es.wikipedia.org/wiki/%C3%8Dndice_de_pobreza_Foster-Greer-Thorbecke](https://es.wikipedia.org/wiki/%C3%8Dndice_de_pobreza_Foster-Greer-Thorbecke)
*   Desigualdad
[https://es.wikipedia.org/wiki/Coeficiente_de_Gini](https://es.wikipedia.org/wiki/Coeficiente_de_Gini)
[https://es.wikipedia.org/wiki/Curva_de_Lorenz](https://es.wikipedia.org/wiki/Curva_de_Lorenz)  
*   Bienestar 
[https://en.wikipedia.org/wiki/Social_welfare_function](https://en.wikipedia.org/wiki/Social_welfare_function)  (Índice de Sen)
*   Polarización
[https://cran.r-project.org/web/packages/affluenceIndex/affluenceIndex.pdf](https://cran.r-project.org/web/packages/affluenceIndex/affluenceIndex.pdf) (Wolfson)
*   Concentración
    [https://es.wikipedia.org/wiki/%C3%8Dndice_de_Herfindahl](https://es.wikipedia.org/wiki/%C3%8Dndice_de_Herfindahl)


    Estos indicadores son independientes, así que se pueden implementar solo algunos en una primera etapa.


    Puede existir la posibilidad de que exista un método genérico y con un parámetro se le pide el indicador (puede ser un vector de indicadores)


    Otro método puede aplicarse para computar la diferencia entre los indicadores de dos distribuciones (si son los mismos individuos -tienen un id-, se pueden calcular matrices de transición pobre-no pobre, por ej.)


## Escala de equivalencia
Permiten calcular el ingreso equivalente del hogar a partir de los ingresos individuales (los indicadores se aplican sobre ambas medidas). Existen diferentes criterios, por lo general se requiere la edad y sexo del individio.
[https://www.ine.es/DEFIne/es/concepto.htm?c=5228&op=30458&p=1&n=20](https://www.ine.es/DEFIne/es/concepto.htm?c=5228&op=30458&p=1&n=20)

## E/S

Ver si es necesario definir un formato de lectura de archivo (una tabla con nombres de las variables predefinidos).
Ver qué datos de la web se pueden cargar (también se podría poner una base de datos real o inventada dentro de la librería)

## Simulación 

El usuario puede utilizar otros paquetes para simular una distribución dada y calcular los indicadores. Ver si tiene sentido vincularlas. 

También podría indicar el indicador (valor del gini, por ej.) y pedirle que le genere una distribución de cierta familia que se corresponda con el indicador.

## Econometría (opcional)

Si los datos son de una muestra se pueden calcular intervalos de confianza para los indicadores.



## Clase


**Variable principal**


Los indicadores en principio tomarían una sola variable principal (habitualmente es el ingreso). Existen indicadores multidimensionales (pobreza multidimensional, etc.) pero son cálculos menos generales.


**Parámetros**


Algunos indicadores requieren parámetros (línea de pobreza, por ej)


**Variables auxiliares**


Son opcionales, cambian el procedimiento del algoritmo.


*   Frecuencia (se cargan datos tabulados). El algoritmo debe poder procesarlo
*   Características. Permiten descomponer el cálculo del indicador. Por ejemplo, calcular la pobreza por provincia y luego la nacional como un promedio ponderado.
*   Escala de equivalencia. Requieren una variable con la edad y otra con el sexo. Y un id que indique hogar 


## Otras Implementaciones

Los investigadores que se dedican a estos temas usan principalmente R (ver libreria ineq) y Stata.

**Python**
[http://www.poorcity.richcity.org/oei/](http://www.poorcity.richcity.org/oei/)  (algoritmos)
[https://github.com/mmngreco/IneqPy](https://github.com/mmngreco/IneqPy)

**R**
[https://cran.r-project.org/web/packages/ineq/ineq.pdf](https://cran.r-project.org/web/packages/ineq/ineq.pdf)
[https://cran.r-project.org/web/packages/affluenceIndex/affluenceIndex.pdf](https://cran.r-project.org/web/packages/affluenceIndex/affluenceIndex.pdf)

**Stata**
[https://www.stata.com/manuals/rinequality.pdf](https://www.stata.com/manuals/rinequality.pdf)


## Referencias

* Handbook on Poverty and Inequality
[https://openknowledge.worldbank.org/bitstream/handle/10986/11985/9780821376133.pdf](https://openknowledge.worldbank.org/bitstream/handle/10986/11985/9780821376133.pdf)
* POBREZA Y DESIGUALDAD EN AMÉRICA LATINA
[https://www.cedlas.econo.unlp.edu.ar/wp/wp-content/uploads/Pobreza_desigualdad_-America_Latina.pdf](https://www.cedlas.econo.unlp.edu.ar/wp/wp-content/uploads/Pobreza_desigualdad_-America_Latina.pdf)
* [https://www.cepal.org/es/publicaciones/4740-enfoques-la-medicion-la-pobreza-breve-revision-la-literatura](https://www.cepal.org/es/publicaciones/4740-enfoques-la-medicion-la-pobreza-breve-revision-la-literatura)
* [https://www.cepal.org/es/publicaciones/4788-consideraciones-indice-gini-medir-la-concentracion-ingreso](https://www.cepal.org/es/publicaciones/4788-consideraciones-indice-gini-medir-la-concentracion-ingreso)

