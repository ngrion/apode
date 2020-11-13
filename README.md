# Apode

<!--- These are examples. See https://shields.io for others or to customize this set of shields. You might want to include dependencies, project status and licence info here --->

[![Build Status](https://travis-ci.org/ngrion/apode.svg?branch=master)](https://travis-ci.org/ngrion/apode)
[![Documentation Status](https://readthedocs.org/projects/apode/badge/?version=latest)](https://apode.readthedocs.io/en/latest/?badge=latest)
[![Coverage Status](https://coveralls.io/repos/github/ngrion/apode/badge.svg?branch=master)](https://coveralls.io/github/ngrion/apode?branch=master) 
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python 3.6+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/release/python-370/)
[![GitHub issues](https://img.shields.io/github/issues/ngrion/apode)](https://github.com/ngrion/apode/issues)
[![GitHub forks](https://img.shields.io/github/forks/ngrion/apode)](https://github.com/ngrion/apode/network)
[![GitHub stars](https://img.shields.io/github/stars/ngrion/apode)](https://github.com/ngrion/apode/stargazers)

Apode is a package that contains a set of indicators that are applied in economic analysis. It contains measures of poverty, inequality, polarization, wealth and concentration.


## Requirements
You need Python 3.8 to run Apode.


## Installation
Clone this repo and then inside the local directory execute

    $ pip install -e .


## Features

Objects are created using:

    ad = ApodeData(DataFrame,varx)
    
Where varx is the name of a column in the dataframe.

Methods that calculate indicators:
   
    ad.poverty(method,*args)    
    ad.ineq(method,*args)
    ad.welfare(method,*args) 
    ad.polarization(method,*args)
    ad.concentration(method,*args)
 
Graphical representations:

    ad.plot.hist()
    ad.plot.tip(**kwargs)
    ad.plot.lorenz(**kwargs)
    ad.plot.pen(**kwargs)
    


## Using Apode

To use Apode, follow these steps:


```python
x = [23, 10, 12, 21, 4, 8, 19, 15, 11, 9]
df = pd.DataFrame({'x':x})
ad = ApodeData(df1, varx="x") 
```

```python
pline = 10 # Poverty line
p = ad.poverty('headcount',pline=pline)
p
```

For more examples, please refer to the [Tutorial](https://apode.readthedocs.io/en/latest/Tutorial.html).

[Documentation](https://apode.readthedocs.io/en/latest/).


## Contributors

Thanks to the following people who have contributed to this project:

* [@ngrion](https://github.com/ngrion) 
* [@sofisappia](https://github.com/sofisappia) 


## Support

If you want to contact me you can reach me at <ngrion@gmail.com>.
If you have issues please report them as a issue [here](https://github.com/ngrion/apode/issues).


## License

Distributed under the MIT License. See [LICENSE](https://github.com/ngrion/apode/blob/master/LICENSE.txt) for more information.


## References

* Cowell, F. (2011) Measuring Inequality. London School of Economics Perspectives in Economic Analysis. 3rd ed. Edición. Oxford University Press
http://darp.lse.ac.uk/papersDB/Cowell_measuringinequality3.pdf
* Cowell, F. (2016) “Inequality and Poverty Measures”, in Oxford Handbook of Well-Being And Public Policy, edited by Matthew D. Adler and Marc Fleurbaey 
* Haughton, J. and S. Khandker (2009). Handbook on Poverty + Inequality. World Bank Training Series. https://openknowledge.worldbank.org/bitstream/handle/10986/11985/9780821376133.pdf
* POBREZA Y DESIGUALDAD EN AMÉRICA LATINA. https://www.cedlas.econo.unlp.edu.ar/wp/wp-content/uploads/Pobreza_desigualdad_-America_Latina.pdf
* Araar Abdelkrim and Jean-Yves Duclos (2007). "DASP: Distributive Analysis  Stata Package", PEP, World Bank, UNDP and Université Laval. http://dasp.ecn.ulaval.ca/


