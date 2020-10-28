# APODE

<!--- These are examples. See https://shields.io for others or to customize this set of shields. You might want to include dependencies, project status and licence info here --->
![GitHub repo size](https://img.shields.io/github/repo-size/ngrion/README.md)
![GitHub contributors](https://img.shields.io/github/contributors/ngrion/README.md)
![GitHub stars](https://img.shields.io/github/stars/ngrion/README.md?style=social)
![GitHub forks](https://img.shields.io/github/forks/ngrion/README.md?style=social)

Apode is a package that contains a set of indicators that are applied in economic analysis. It contains measures of poverty, inequality, polarization, wealth and concentration.


## Prerequisites

Before you begin, ensure you have met the following requirements:
<!--- These are just example requirements. Add, duplicate or remove as required --->
* You have installed the latest version of `<coding_language/dependency/requirement_1>`
* You have read `<guide/link/documentation_related_to_project>`.

## Installing Apode

To install Apode, follow these steps:

Linux and macOS:
```
$ pip install apode
```

Windows:
```
$ pip install apode
```

## ApodeData Class

Objects are created using:

    df = ApodeData(DataFrame,varx)
    
Where varx is the name of a column in the dataframe.

Methods that calculate indicators:
   
    df.poverty(method,*args)    
    df.ineq(method,*args)
    df.welfare(method,*args) 
    df.polar(method,*args)
    df.conc(method,*args)
 
Graphical representations:

    df.tip(*args,**kwargs)
    df.lorenz(*args,**kwargs)
    df.pen(*args,**kwargs)
    

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

For more examples, please refer to the [Tutorial](https://github.com/ngrion/apode/blob/master/apode/doc/Tutorial.md).


## Contributing to Apode
<!--- If your README is long or you have some specific process or steps you want contributors to follow, consider creating a separate CONTRIBUTING.md file--->
To contribute to Apode, follow these steps:

1. Fork this repository.
2. Create a branch: `git checkout -b <branch_name>`.
3. Make your changes and commit them: `git commit -m '<commit_message>'`
4. Push to the original branch: `git push origin <project_name>/<location>`
5. Create the pull request.

Alternatively see the GitHub documentation on [creating a pull request](https://help.github.com/en/github/collaborating-with-issues-and-pull-requests/creating-a-pull-request).

## Contributors

Thanks to the following people who have contributed to this project:

* [@ngrion](https://github.com/ngrion) 
* [@sofisappia](https://github.com/sofisappia) 


You might want to consider using something like the [All Contributors](https://github.com/all-contributors/all-contributors) specification and its [emoji key](https://allcontributors.org/docs/en/emoji-key).


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



