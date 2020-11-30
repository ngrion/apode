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

import attr

import pandas as pd

from .concentration import ConcentrationMeasures
from .inequality import InequalityMeasures
from .plots import PlotAccsessor
from .polarization import PolarizationMeasures
from .poverty import PovertyMeasures
from .welfare import WelfareMeasures


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
    poverty = attr.ib(
        init=False, default=attr.Factory(PovertyMeasures, takes_self=True)
    )
    inequality = attr.ib(
        init=False, default=attr.Factory(InequalityMeasures, takes_self=True)
    )
    polarization = attr.ib(
        init=False, default=attr.Factory(PolarizationMeasures, takes_self=True)
    )
    concentration = attr.ib(
        init=False,
        default=attr.Factory(ConcentrationMeasures, takes_self=True),
    )
    welfare = attr.ib(
        init=False, default=attr.Factory(WelfareMeasures, takes_self=True)
    )
    plot = attr.ib(
        init=False, default=attr.Factory(PlotAccsessor, takes_self=True)
    )

    @income_column.validator
    def _validate_income_column(self, name, value):
        if value not in self.data.columns:
            raise ValueError()
    @data.validator
    def _validate_data(self, name, value):
        if self.data.empty:
            raise ValueError()

    def __getattr__(self, aname):
        """Apply DataFrame method."""
        return getattr(self.data, aname)

    def __getitem__(self, slice):
        """Apply Slice method."""
        data = self.data.__getitem__(slice)
        if not hasattr(data, self.income_column):
            raise AttributeError(
                f"Cannot take column {self.income_column} "
                "from ApodeData object"
            )
        return ApodeData(data, income_column=self.income_column)

    def __repr__(self):
        """Apply Display method."""
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
        footer = f"ApodeData(income_column='{income_column}') - {rows} x \
            {columns}"
        return footer

    def __dir__(self):
        """Allow access to methods and attributes of the dataframe."""
        return super().__dir__() + dir(self.data)
