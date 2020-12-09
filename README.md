# Apode


![logo](https://raw.githubusercontent.com/ngrion/apode/master/res/logo.png)

[![Build Status](https://travis-ci.com/ngrion/apode.svg?branch=master)](https://travis-ci.org/ngrion/apode)
[![Documentation Status](https://readthedocs.org/projects/apode/badge/?version=latest)](https://apode.readthedocs.io/en/latest/?badge=latest)
[![Coverage Status](https://coveralls.io/repos/github/ngrion/apode/badge.svg?branch=master)](https://coveralls.io/github/ngrion/apode?branch=master) 
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python 3.6+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/release/python-370/)
[![GitHub issues](https://img.shields.io/github/issues/ngrion/apode)](https://github.com/ngrion/apode/issues)
[![GitHub forks](https://img.shields.io/github/forks/ngrion/apode)](https://github.com/ngrion/apode/network)
[![GitHub stars](https://img.shields.io/github/stars/ngrion/apode)](https://github.com/ngrion/apode/stargazers)
[![https://github.com/leliel12/diseno_sci_sfw](https://img.shields.io/badge/DiSoftCompCi-FAMAF-ffda00)](https://github.com/leliel12/diseno_sci_sfw)

Apode is a package that contains a set of indicators that are applied in economic analysis. It contains measures of poverty, inequality, polarization, welfare and concentration.




## Requirements
You need Python 3.8 to run Apode.

## Code Repository & Issues
https://github.com/quatrope/pycf3

## Basic Install

Execute

```console
$ pip install apode
```

## Development Install


Clone this repo and install with pip

```console
$ git clone https://github.com/ngrion/apode.git
$ cd pycf3
$ pip install -e .
```


## Features

Objects are created using:

```pycon
>>> ad = ApodeData(DataFrame, income_column)
```
    
Where income_column is the name of the desired analysis column in the dataframe.

Methods that calculate indicators:

```pycon
>>> ad.poverty(method,*args)    
>>> ad.ineq(method,*args)
>>> ad.welfare(method,*args) 
>>> ad.polarization(method,*args)
>>> ad.concentration(method,*args)
```
 
Graphical representations:

```pycon
>>> ad.plot.hist()
>>> ad.plot.tip(**kwargs)
>>> ad.plot.lorenz(**kwargs)
>>> ad.plot.pen(**kwargs)
```

For examples on how to use apode, please refer to the [Tutorial](https://apode.readthedocs.io/en/latest/Tutorial.html).


## Contributors

Thanks to the following people who have contributed to this project:

* [@ngrion](https://github.com/ngrion) 
* [@sofisappia](https://github.com/sofisappia) 


## Support

If you want to contact me you can reach me at <ngrion@gmail.com>.
If you have issues please report them as a issue [here](https://github.com/ngrion/apode/issues).


## License

Distributed under the MIT License. See [LICENSE](https://github.com/ngrion/apode/blob/master/LICENSE.txt) for more information.



