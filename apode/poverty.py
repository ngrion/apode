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

"""Poverty measures for Apode."""

# =============================================================================
# IMPORTS
# =============================================================================


import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
import attr


# =============================================================================
# FUNCTIONS
# =============================================================================


@attr.s(frozen=True)
class PovertyMeasures:
    """Poverty Measures.

    The following poverty measures are implemented:
    - headcount: Headcount Index
    - gap: Poverty gap Index
    - severity: Poverty Severity Index
    - fgt: Foster–Greer–Thorbecke Indices
    - sen: Sen Index
    - sst: Sen-Shorrocks-Thon Index
    - watts: Watts Index
    - cuh: Clark, Ulph and Hemming Indices
    - takayama: Takayama Index
    - kakwani: Kakwani Indices
    - thon: Thon Index
    - bd: Blackorby and Donaldson Indices
    - hagenaars: Hagenaars Index
    - chakravarty: Chakravarty Indices

    Parameters
    ----------
    method: String
        Poverty measure.
    kwargs: Any
        Method specific parameters.

    """

    idf = attr.ib()

    def __call__(self, method=None, **kwargs):
        """Return the ApodeData object."""
        method = "headcount" if method is None else method
        method_func = getattr(self, method)
        return method_func(**kwargs)

    def headcount(self, pline=None):
        """Headcount index.

        The headcount index measures the proportion of the population that
        counted as poor.

        Parameters
        ----------
        pline: float, optional(default=None)
            Poverty line. If is None then pline = 0.5*median(y).

        Return
        ------
        out: float
            Headcount index measures.

        """
        y = self.idf.data[self.idf.varx].values
        pline = _get_pline(y, pline)
        n = len(y)
        ys = np.sort(y)
        q = np.sum(ys < pline)
        return q / n

    def gap(self, pline=None):
        """Poverty gap index.

        The poverty gap index adds up the extent to which individuals
        on average fall below the poverty line, and expresses it as
        a percentage of the poverty line.

        Parameters
        ----------
        pline: float, optional(default=None)
            Poverty line. If is None then pline = 0.5*median(y).

        Return
        ------
        out: float
            Poverty gap index measures.

        """
        y = self.idf.data[self.idf.varx].values
        pline = _get_pline(y, pline)
        n = len(y)
        ys = np.sort(y)
        q = np.sum(ys < pline)
        yp = ys[0:q]
        br = (pline - yp) / pline
        return np.sum(br) / n

    def severity(self, pline=None):
        """Squared Poverty Gap (Poverty Severity) Index.

        To construct a measure of poverty that takes into account inequality
        among the poor, some researchers use the squared poverty gap index.
        This is simply a weighted sum of poverty gaps (as a proportion of the
        poverty line), where the weights are the proportionate poverty gaps
        themselves

        Parameters
        ----------
        pline: float, optional(default=None)
            Poverty line. If is None then pline = 0.5*median(y).

        Return
        ------
        out: float
            Poverty Severity Index measures.

        """
        y = self.idf.data[self.idf.varx].values
        pline = _get_pline(y, pline)
        n = len(y)
        ys = np.sort(y)
        q = np.sum(ys < pline)
        yp = ys[0:q]
        br = np.power((pline - yp) / pline, 2)
        return np.sum(br) / n

    def fgt(self, pline=None, alpha=0):
        """Foster–Greer–Thorbecke Indices.

        When parameter α = 0, P0 is simply the headcount index. When α = 1,
        the index is the poverty gap index P1, and when α is set equal
        to 2, P2 is the poverty severity index.
        A α se le conoce con el nombre de parámetro de aversión a la pobreza y,
        por tanto, cuanto mayor sea α, más énfasis se le da al más pobre de
        los pobres.

        Parameters
        ----------
        pline: float, optional(default=None)
            Poverty line. If is None then pline = 0.5*median(y).

        alpha: float, optional(default=0)
            Aversion poverty parameter.

        Return
        ------
        out: float
            Foster–Greer–Thorbecke Indices measures.

        """
        y = self.idf.data[self.idf.varx].values
        pline = _get_pline(y, pline)
        n = len(y)
        ys = np.sort(y)
        q = np.sum(ys < pline)
        yp = ys[0:q]
        if alpha < 0:
            raise ValueError(f"'alpha' must be >= 0. Found '{alpha}'")
        elif alpha == 0:
            return q / n
        elif alpha == 1:
            br = (pline - yp) / pline
            return np.sum(br) / n
        br = np.power((pline - yp) / pline, alpha)
        return np.sum(br) / n

    def sen(self, pline=None):
        """Sen Index.

        Sen (1976) proposed an index that seeks to combine the effects of the
        number of poor, the depth of their poverty, and the distribution
        poverty within the group.

        Parameters
        ----------
        pline: float, optional(default=None)
            Poverty line. If is None then pline = 0.5*median(y).

        Return
        ------
        out: float
            Sen Index Index measures.

        """
        p0 = self.headcount(pline=pline)
        p1 = self.gap(pline=pline)
        gp = self.idf.inequality.gini()
        return p0 * gp + p1 * (1 - gp)

    def sst(self, pline=None):
        """Sen-Shorrocks-Thon Index Index.

        The Sen index has been modified by others, and one of the
        most attractive versions is the Sen-Shorrocks-Thon (SST)
        index. One strength of the SST index is that it can help
        give a good sense of the sources of change in poverty over
        time. This is because the index can be decomposed.

        Parameters
        ----------
        pline: float, optional(default=None)
            Poverty line. If is None then pline = 0.5*median(y).

        Return
        ------
        out: float
            Sen-Shorrocks-Thon Index measure.

        """
        p0 = self.headcount(pline=pline)
        p1 = self.gap(pline=pline)
        gp = self.idf.inequality.gini()
        return p0 * p1 * (1 + gp)

    def watts(self, pline=None):
        """Watts index.

        Harold Watts (1968) propuso la siguiente medida de
        pobreza sensible a la distribución de rentas.

        Parameters
        ----------
        pline: float, optional(default=None)
            Poverty line. If is None then pline = 0.5*median(y).

        Return
        ------
        out: float
            Watts Index measure.

        """
        y = self.idf.data[self.idf.varx].values
        pline = _get_pline(y, pline)
        n = len(y)
        ys = np.sort(y)
        q = np.sum(ys < pline)
        yp = ys[0:q]
        return sum(np.log(pline / yp)) / n

    def cuh(self, pline=None, alpha=0):
        """Clark, Ulph and Hemming index.

        Clark, Hemming y Ulph (1981) proponen utilizar en la medida
        de pobreza de Sen, la medida de Atkinson en lugar del índice
        de Gini de los pobres.

        Parameters
        ----------
        pline: float, optional(default=None)
            Poverty line. If is None then pline = 0.5*median(y).

        alpha: float, optional(default=0)
            Atkinson parameter.

        Return
        ------
        out: float
            Clark, Ulph and Hemming Indices measures.

        """
        y = self.idf.data[self.idf.varx].values
        pline = _get_pline(y, pline)
        n = len(y)
        ys = np.sort(y)
        q = np.sum(ys < pline)
        yp = ys[0:q]
        if (alpha < 0) or (alpha > 1):
            raise ValueError(f"'alpha' must be in [0,1]. Found '{alpha}'")
        if alpha == 0:
            return 1 - np.power(np.product(yp / pline) / n, 1 / n)
        else:
            return 1 - np.power(
                (sum(np.power(yp / pline, alpha)) + (n - q)) / n, 1 / alpha
            )

    def takayama(self, pline=None):
        """Takayama Index.

        Takayama (1979) define su medida de pobreza calculando el índice
        de Gini de la distribución censurada por la línea de pobreza.

        Parameters
        ----------
        pline: float, optional(default=None)
            Poverty line. If is None then pline = 0.5*median(y).

        Return
        ------
        out: float
            Takayama Index measure.

        """
        y = self.idf.data[self.idf.varx].values
        pline = _get_pline(y, pline)
        n = len(y)
        ys = np.sort(y)
        q = np.sum(ys < pline)
        if q == 0:
            return 0  # CHECK THIS!!
        yp = ys[0:q]
        u = (yp.sum() + (n - q) * pline) / n
        a = 0
        for i in range(0, q):
            a = a + (n - i + 1) * ys[i]
        for i in range(q, n):
            a = a + (n - i + 1) * pline
        if u * n * n == 0:
            return 0  # to avoid NaNs for zero division error
        return 1 + 1 / n - (2 / (u * n * n)) * a

    # Kakwani Index
    def kakwani(self, pline=None, alpha=2):
        """Kakwani Indices.

        La familia de Kakwani (1980) que pondera los déficit mediante
        una potencia del número de orden que ocupa cada individuo
        dentro del subgrupo de pobres. El parámetro α identifica una
        cierta “aversión” al lugar ocupado en la sociedad.

        Parameters
        ----------
        pline: float, optional(default=None)
            Poverty line. If is None then pline = 0.5*median(y).

        alpha: float, optional(default=2)
            Aversion parameter.

        Return
        ------
        out: float
            Kakwani Indices measures.

        """
        y = self.idf.data[self.idf.varx].values
        pline = _get_pline(y, pline)
        n = len(y)
        ys = np.sort(y)
        q = np.sum(ys < pline)
        # yp = ys[0:q] # unused, commenting it out (delete?)
        # alpha = 2.0  # elegible
        a = 0.0
        u = 0.0
        for i in range(0, q):
            f = np.power(q - i + 2, alpha)  # ver +2
            a = a + f
            u = u + f * (pline - ys[i])
        if u == 0:
            return 0  # to avoid NaNs for zero division error
        return (q / (n * pline * a)) * u

    def thon(self, pline=None):
        """Thon Index.

        La diferencia entre esta medida (Thon,1979) y la de Sen radica
        en la función de ponderación. Aquí se pondera el individuo pobre
        por el lugar que ocupa dentro de toda la población, y no solo
        respecto a los pobres.

        Parameters
        ----------
        pline: float, optional(default=None)
            Poverty line. If is None then pline = 0.5*median(y).

        Return
        ------
        out: float
            Thon Index measure.

        """
        y = self.idf.data[self.idf.varx].values
        pline = _get_pline(y, pline)
        if pline == 0:
            return 0.0
        n = len(y)
        ys = np.sort(y)
        q = np.sum(ys < pline)
        # yp = ys[0:q] # unused, commenting it out (delete?)
        u = 0
        for i in range(0, q):
            u = u + (n - i + 1) * (pline - ys[i])
        return (2 / (n * (n + 1) * pline)) * u

    def bd(self, pline=None, alpha=2):
        """Blackorby and Donaldson Indices.

        Blackorby y Donaldson (1980) proponen una medida de pobreza de tipo
        normativo.

        Parameters
        ----------
        pline: float, optional(default=None)
            Poverty line. If is None then pline = 0.5*median(y).

        alpha: float, optional(default=2)
            Aversion parameter. (ver)

        Return
        ------
        out: float
            Kakwani Indices measures.

        """
        y = self.idf.data[self.idf.varx].values
        pline = _get_pline(y, pline)
        n = len(y)
        ys = np.sort(y)
        q = np.sum(ys < pline)
        if q == 0:
            return 0  # CHECK IF CORRECT
        yp = ys[0:q]
        u = yp.sum() / q
        # atkp = atkinson(yp, alpha)
        # gp = self.idf.inequality.gini()
        atkp = self.idf.inequality.atkinson(alpha=alpha)
        yedep = u * (1 - atkp)
        return (q / n) * (pline - yedep) / pline

    def hagenaars(self, pline=None):
        """Hagenaars Index.

        Hagenaars (1984) para obtener la medida de pobreza considera
        la función de evaluación social de la renta como V(x) = ln(x).

        Parameters
        ----------
        pline: float, optional(default=None)
            Poverty line. If is None then pline = 0.5*median(y).

        Return
        ------
        out: float
            Hagenaars Index measure.

        """
        y = self.idf.data[self.idf.varx].values
        pline = _get_pline(y, pline)
        n = len(y)
        ys = np.sort(y)
        q = np.sum(ys < pline)
        if q == 0:
            return 0  # check this!!
        yp = ys[0:q]
        ug = np.exp(sum(np.log(yp)) / q)  # o normalizar con el maximo
        return (q / n) * ((np.log(pline) - np.log(ug)) / np.log(pline))

    def chakravarty(self, pline=None, alpha=0.5):
        """Chakravarty Indices.

        Chakravarty (1983) es una medida ética de pobreza. El índice de pobreza
        se obtiene, entonces, como la suma normalizada de las carencias de
        utilidad de los pobres.

        Parameters
        ----------
        pline: float, optional(default=None)
            Poverty line. If is None then pline = 0.5*median(y).

        alpha: float, optional(default=0.5)
            Aversion parameter. (ver)

        Return
        ------
        out: float
            Chakravarty Indices measures.

        """
        y = self.idf.data[self.idf.varx].values
        pline = _get_pline(y, pline)
        if (alpha <= 0) or (alpha >= 1):
            raise ValueError(f"'alpha' must be in (0,1). Found '{alpha}'")
        n = len(y)
        ys = np.sort(y)
        q = np.sum(ys < pline)
        yp = ys[0:q]
        return sum(1 - np.power(yp / pline, alpha)) / n


def _get_pline(y, pline=None):
    """Check/calcule poverty line."""
    if pline is None:
        return 0.5*np.median(y)
    elif pline < 0:
        raise ValueError(f"'pline' must be >= 0. Found '{pline}'")
    else:
        return pline
