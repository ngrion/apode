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
from .plots import PlotAccsesor


# =============================================================================
# MAIN CLASS
# =============================================================================

@attr.s(frozen=True, repr=False)
class ApodeData:
    """Poverty and Inequality Analysis in Python.

    Apode is a package that contains a set of indicators that are applied in
    economic analysis. It contains measures of:
    - poverty
    - inequality
    - welfare
    - polarization
    - concentration

    Parameters
    ----------
    data : dataframe
        Dataset to be analiced.

    income_column : str
        Column name

    Attributes
    ----------
    data, income_column : see Parameters

    """

    data = attr.ib(converter=pd.DataFrame)
    income_column = attr.ib()
    poverty = attr.ib(init=False)
    inequality = attr.ib(init=False)
    polarization = attr.ib(init=False)
    concentration = attr.ib(init=False)
    welfare = attr.ib(init=False)
    plot = attr.ib(init=False)

    @income_column.validator
    def _validate_income_column(self, name, value):
        if value not in self.data.columns:
            raise ValueError()

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

    @plot.default
    def _plot_default(self):
        return PlotAccsesor(idf=self)

    def __getattr__(self, aname):
        """Apply DataFrame method."""
        return getattr(self.data, aname)

    # def __repr__(self):
    #     df_body = repr(self.data).splitlines()
    #     df_dim = list(self.data.shape)
    #     sdf_dim = f"[{df_dim[0]} x {df_dim[1]}]"
    #     if len(df_body) <= df_dim[0]:  # si df_body está recortado
    #         df_body = df_body[:-2]     # se elimina descripción final
    #     fotter = (f"\nApodeData(income_column='{self.income_column}', "
    #               f"{sdf_dim})")
    #     apode_data_repr = "\n".join(df_body + [fotter])
    #     return apode_data_repr

    def __repr__(self):
        with pd.option_context("display.show_dimensions", False):
            df_body = repr(self.data).splitlines()
        footer = self._get_footer()
        brepr = "\n".join(df_body + [footer])
        return brepr

    def _repr_html_(self):
        with pd.option_context("display.show_dimensions", False):
            df_html = self.data._repr_html_()
        ad_id = id(self)
        footer = self._get_footer(html=True)
        parts = [
            f'<div class="apode-data-container" id={ad_id}>',
            df_html,
            footer,
            "</div>",
        ]
        html = "".join(parts)
        return html

    def _get_footer(self, html=None):
        income_column = self.income_column
        if html is True:
            income_column = f"<i>{income_column}</i>"
        rows = f"{self.data.shape[0]} rows"
        columns = f"{self.data.shape[1]} columns"
        footer = (
            f"ApodeData(income_column='{income_column}') - {rows} x {columns}"
        )
        return footer

    def __getitem__(self, slice):
        df = self.data.__getitem__(slice)
        return ApodeData(df, income_column=self.income_column)

    def __dir__(self):
        """Permite acceder a metodos y atributos del dataframe subyacente"""
        return super().__dir__() + dir(self.data)
