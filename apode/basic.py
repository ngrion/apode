#!/usr/bin/env python
# -*- coding: utf-8 -*-

# This file is part of the
#   Apode Project (https://github.com/ngrion/apode).
# Copyright (c) 2020, Néstor Grión and Sofía Sappia
# License: MIT
#   Full Text: https://github.com/ngrion/apode/blob/master/LICENSE.txt

# =============================================================================
# DOCS
# =============================================================================

"""ApodeData class for Apode."""


# =============================================================================
# IMPORTS
# =============================================================================


import pandas as pd

import attr

from .poverty import PovertyMeasures
from .inequality import InequalityMeasures
from .welfare import WelfareMeasures
from .polarization import PolarizationMeasures
from .concentration import ConcentrationMeasures


# =============================================================================
# MAIN CLASS
# =============================================================================


@attr.s(frozen=True)
class ApodeData:
    """Poverty and Inequality Analysis in Python.
    Apode es un paquete que contiene un conjunto de indicadores que se aplican
    en el análisis económico. Contiene algoritmos referidos a:
    - poverty: .
    - inequality:
    - welfare:
    - polarization:
    - concentration:
    Parameters
    ----------
    data: dataframe, shape(n,k)
        The n data points of dimension k to be analiced.
    varx: Column name
        Debe existir en el dataframe
    Attributes
    ----------
    ninguno: int
        Ninguno?.
    """

    data = attr.ib(converter=pd.DataFrame)
    varx = attr.ib()
    poverty = attr.ib(init=False)
    inequality = attr.ib(init=False)
    polarization = attr.ib(init=False)
    concentration = attr.ib(init=False)
    welfare = attr.ib(init=False)

    @poverty.default
    def _poverty_default(self):
        return PovertyMeasures(idf=self)

    @inequality.default
    def _inequality_default(self):
        return InequalityMeasures(idf=self)

    @polarization.default
    def _polarization_default(self):
        return PolarizationMeasures(idf=self)

    @concentration.default
    def _concentration_default(self):
        return ConcentrationMeasures(idf=self)

    @welfare.default
    def _welfare_default(self):
        return WelfareMeasures(idf=self)

    @varx.validator
    def _validate_varx(self, name, value):
        if value not in self.data.columns:
            raise ValueError()

    def __getattr__(self, aname):
        return getattr(self.data, aname)
