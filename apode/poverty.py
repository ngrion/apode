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
import attr

import numpy as np


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
    method : String
        Poverty measure.
    **kwargs
        Arbitrary keyword arguments.

    """

    idf = attr.ib()

    def __call__(self, method=None, **kwargs):
        """Return the ApodeData object."""
        method = "headcount" if method is None else method
        method_func = getattr(self, method)
        return method_func(**kwargs)

    def headcount(self, pline=None, factor=1.0, q=None):
        """Headcount index.

        The headcount index measures the proportion of the population that
        counted as poor.

        Parameters
        ----------
        pline : optional(default=None)
            Absolute poverty line if pline is float.
            Relative poverty line if pline is 'median', 'quantile' or 'mean'
            If pline is None then pline = 0.5*median(y).

        factor : float, optional(default=1.0)
            Factor in pline = factor*stat

        q : float, optional(default=None)
            Cuantil q if pline is'quantile'

        Return
        ------
        out: float
            Index measure.

        References
        ----------
        .. Haughton, J., and Khandker, S. R. (2009). Handbook on poverty
           and inequality. Washington, DC: World Bank.

        """
        y = self.idf.data[self.idf.income_column].values
        pline = _get_pline(y, pline, factor, q)
        n = len(y)
        ys = np.sort(y)
        q = np.sum(ys < pline)
        return q / n

    def gap(self, pline=None, factor=1.0, q=None):
        """Poverty gap index.

        The poverty gap index adds up the extent to which individuals
        on average fall below the poverty line, and expresses it as
        a percentage of the poverty line.

        Parameters
        ----------
        pline : optional(default=None)
            Absolute poverty line if pline is float.
            Relative poverty line if pline is 'median', 'quantile' or 'mean'
            If pline is None then pline = 0.5*median(y).

        factor : float, optional(default=1.0)
            Factor in pline = factor*stat

        q : float, optional(default=None)
            Cuantil q if pline is'quantile'

        Return
        ------
        out: float
            Index measure.

        References
        ----------
        .. Haughton, J., and Khandker, S. R. (2009). Handbook on poverty
           and inequality. Washington, DC: World Bank.

        """
        y = self.idf.data[self.idf.income_column].values
        pline = _get_pline(y, pline, factor, q)
        n = len(y)
        ys = np.sort(y)
        q = np.sum(ys < pline)
        yp = ys[0:q]
        br = (pline - yp) / pline
        return np.sum(br) / n

    def severity(self, pline=None, factor=1.0, q=None):
        """Squared Poverty Gap (Poverty Severity) Index.

        To construct a measure of poverty that takes into account inequality
        among the poor, some researchers use the squared poverty gap index.
        This is simply a weighted sum of poverty gaps (as a proportion of the
        poverty line), where the weights are the proportionate poverty gaps
        themselves

        Parameters
        ----------
        pline : optional(default=None)
            Absolute poverty line if pline is float.
            Relative poverty line if pline is 'median', 'quantile' or 'mean'
            If pline is None then pline = 0.5*median(y).

        factor : float, optional(default=1.0)
            Factor in pline = factor*stat

        q : float, optional(default=None)
            Cuantil q if pline is'quantile'

        Return
        ------
        out: float
            Index measure.

        References
        ----------
        .. Haughton, J., and Khandker, S. R. (2009). Handbook on poverty
           and inequality. Washington, DC: World Bank.

        """
        y = self.idf.data[self.idf.income_column].values
        pline = _get_pline(y, pline, factor, q)
        n = len(y)
        ys = np.sort(y)
        q = np.sum(ys < pline)
        yp = ys[0:q]
        br = np.power((pline - yp) / pline, 2)
        return np.sum(br) / n

    def fgt(self, pline=None, alpha=0, factor=1.0, q=None):
        """Foster–Greer–Thorbecke Indices.

        When parameter α = 0, P0 is simply the headcount index. When α = 1,
        the index is the poverty gap index P1, and when α is set equal
        to 2, P2 is the poverty severity index.
        A α se le conoce con el nombre de parámetro de aversión a la pobreza y,
        por tanto, cuanto mayor sea α, más énfasis se le da al más pobre de
        los pobres.

        Parameters
        ----------
        pline : optional(default=None)
            Absolute poverty line if pline is float.
            Relative poverty line if pline is 'median', 'quantile' or 'mean'
            If pline is None then pline = 0.5*median(y).

        factor : float, optional(default=1.0)
            Factor in pline = factor*stat

        q : float, optional(default=None)
            Cuantil q if pline is'quantile'

        alpha: float, optional(default=0)
            Aversion poverty parameter.

        Return
        ------
        out: float
            Index measure.

        References
        ----------
        .. Foster, J.E.; Greer, J. y Thorbecke, E. (1984). “A class of
           decomposable poverty measures”. Econometrica. Vol. 52, n 3,
           pp.761–766.

        """
        y = self.idf.data[self.idf.income_column].values
        pline = _get_pline(y, pline, factor, q)
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

    def sen(self, pline=None, factor=1.0, q=None):
        """Sen Index.

        Sen (1976) proposed an index that seeks to combine the effects of the
        number of poor, the depth of their poverty, and the distribution
        poverty within the group.

        Parameters
        ----------
        pline : optional(default=None)
            Absolute poverty line if pline is float.
            Relative poverty line if pline is 'median', 'quantile' or 'mean'
            If pline is None then pline = 0.5*median(y).

        factor : float, optional(default=1.0)
            Factor in pline = factor*stat

        q : float, optional(default=None)
            Cuantil q if pline is'quantile'

        Return
        ------
        out: float
            Index measure.

        References
        ----------
        .. Sen, A. (1976). “Poverty: an ordinal approach to measurement”.
           Econometrica 44(2), pp.219–231.

        """
        p0 = self.headcount(pline=pline, factor=factor, q=q)
        p1 = self.gap(pline=pline, factor=factor, q=q)
        gp = self.idf.inequality.gini()
        return p0 * gp + p1 * (1 - gp)

    def sst(self, pline=None, factor=1.0, q=None):
        """Sen-Shorrocks-Thon Index Index.

        The Sen index has been modified by others, and one of the
        most attractive versions is the Sen-Shorrocks-Thon (SST)
        index. One strength of the SST index is that it can help
        give a good sense of the sources of change in poverty over
        time. This is because the index can be decomposed.

        Parameters
        ----------
        pline : optional(default=None)
            Absolute poverty line if pline is float.
            Relative poverty line if pline is 'median', 'quantile' or 'mean'
            If pline is None then pline = 0.5*median(y).

        factor : float, optional(default=1.0)
            Factor in pline = factor*stat

        q : float, optional(default=None)
            Cuantil q if pline is'quantile'

        Return
        ------
        out: float
            Index measure.

        References
        ----------
        .. Xu, K. (1998). Statistical inference for the Sen-Shorrocks-Thon
           index of poverty intensity. Journal of Income Distribution, 8,
           143-152.

        """
        p0 = self.headcount(pline=pline, factor=factor, q=q)
        p1 = self.gap(pline=pline, factor=factor, q=q)
        gp = self.idf.inequality.gini()
        return p0 * p1 * (1 + gp)

    def watts(self, pline=None, factor=1.0, q=None):
        """Watts index.

        Harold Watts (1968) propuso la siguiente medida de
        pobreza sensible a la distribución de rentas.

        Parameters
        ----------
        pline : optional(default=None)
            Absolute poverty line if pline is float.
            Relative poverty line if pline is 'median', 'quantile' or 'mean'
            If pline is None then pline = 0.5*median(y).

        factor : float, optional(default=1.0)
            Factor in pline = factor*stat

        q : float, optional(default=None)
            Cuantil q if pline is'quantile'

        Return
        ------
        out: float
            Index measure.

        References
        ----------
        .. Watts, H. (1968). “An economic definition of poverty”, en D. P.
           Moynihan. On Understanding Poverty. Basic Books. Inc. New York,
           pp.316–329.

        """
        y = self.idf.data[self.idf.income_column].values
        pline = _get_pline(y, pline, factor, q)
        n = len(y)
        ys = np.sort(y)
        q = np.sum(ys < pline)
        yp = ys[0:q]
        return sum(np.log(pline / yp)) / n

    def cuh(self, pline=None, alpha=0, factor=1.0, q=None):
        """Clark, Ulph and Hemming index.

        Clark, Hemming y Ulph (1981) proponen utilizar en la medida
        de pobreza de Sen, la medida de Atkinson en lugar del índice
        de Gini de los pobres.

        Parameters
        ----------
        pline : optional(default=None)
            Absolute poverty line if pline is float.
            Relative poverty line if pline is 'median', 'quantile' or 'mean'
            If pline is None then pline = 0.5*median(y).

        factor : float, optional(default=1.0)
            Factor in pline = factor*stat

        q : float, optional(default=None)
            Cuantil q if pline is'quantile'

        alpha: float, optional(default=0)
            Atkinson parameter.

        Return
        ------
        out: float
            Index measure.

        References
        ----------
        .. Clark, S.R.; Hemming, R. y Ulph, D. (1981). “On indices for
           the measurement of poverty”. Economic Journal. Vol. 91,
           pp.515–526.

        """
        y = self.idf.data[self.idf.income_column].values
        pline = _get_pline(y, pline, factor, q)
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

    def takayama(self, pline=None, factor=1.0, q=None):
        """Takayama Index.

        Takayama (1979) define su medida de pobreza calculando el índice
        de Gini de la distribución censurada por la línea de pobreza.

        Parameters
        ----------
        pline : optional(default=None)
            Absolute poverty line if pline is float.
            Relative poverty line if pline is 'median', 'quantile' or 'mean'
            If pline is None then pline = 0.5*median(y).

        factor : float, optional(default=1.0)
            Factor in pline = factor*stat

        q : float, optional(default=None)
            Cuantil q if pline is'quantile'

        Return
        ------
        out: float
            Index measure.

        References
        ----------
        .. Takayama, N. (1979). “Poverty, income inequality, and their
           measures: Professor Sen’s axiomatic approach reconsidered”.
           Econometrica. Vol. 47, n 3, pp.747–759.

        """
        y = self.idf.data[self.idf.income_column].values
        pline = _get_pline(y, pline, factor, q)
        n = len(y)
        ys = np.sort(y)
        q = np.sum(ys < pline)
        if q == 0:
            return 0  # CHECK THIS!!
        yp = ys[0:q]
        u = (yp.sum() + (n - q) * pline) / n
        if u * n * n == 0:
            return 0  # to avoid NaNs for zero division error
        i_0q = np.arange(q)
        i_qn = np.arange(q, n)
        a = np.sum(np.dot(n - i_0q + 1, ys[:q])) + np.sum(
            (n - i_qn + 1) * pline
        )  # )
        return 1 + 1 / n - (2 / (u * n * n)) * a

    # Kakwani Index
    def kakwani(self, pline=None, alpha=2, factor=1.0, q=None):
        """Kakwani Indices.

        La familia de Kakwani (1980) que pondera los déficit mediante
        una potencia del número de orden que ocupa cada individuo
        dentro del subgrupo de pobres. El parámetro α identifica una
        cierta “aversión” al lugar ocupado en la sociedad.

        Parameters
        ----------
        pline : optional(default=None)
            Absolute poverty line if pline is float.
            Relative poverty line if pline is 'median', 'quantile' or 'mean'
            If pline is None then pline = 0.5*median(y).

        factor : float, optional(default=1.0)
            Factor in pline = factor*stat

        q : float, optional(default=None)
            Cuantil q if pline is'quantile'

        alpha: float, optional(default=2)
            Aversion parameter.

        Return
        ------
        out: float
            Index measure.

        References
        ----------
        .. Kakwani, Nanak (1980). “On a Class of Poverty Measures”.
           Econometrica, vol.48, n.2, pp.437-446

        """
        y = self.idf.data[self.idf.income_column].values
        pline = _get_pline(y, pline, factor, q)
        n = len(y)
        ys = np.sort(y)
        q = np.sum(ys < pline)
        ii = np.arange(q)
        f = np.power(q - ii + 2, alpha)
        a = np.float(np.sum(f))
        u = np.sum(np.dot(f, pline - ys[:q]))
        if u == 0:
            return 0  # to avoid NaNs for zero division error
        return (q / (n * pline * a)) * u

    def thon(self, pline=None, factor=1.0, q=None):
        """Thon Index.

        La diferencia entre esta medida (Thon,1979) y la de Sen radica
        en la función de ponderación. Aquí se pondera el individuo pobre
        por el lugar que ocupa dentro de toda la población, y no solo
        respecto a los pobres.

        Parameters
        ----------
        pline : optional(default=None)
            Absolute poverty line if pline is float.
            Relative poverty line if pline is 'median', 'quantile' or 'mean'
            If pline is None then pline = 0.5*median(y).

        factor : float, optional(default=1.0)
            Factor in pline = factor*stat

        q : float, optional(default=None)
            Cuantil q if pline is'quantile'

        Return
        ------
        out: float
            Index measure.

        References
        ----------
        .. Thon, D. (1979). “On measuring poverty”. Review of Income
           and Wealth. Vol. 25, pp.429–439.

        """
        y = self.idf.data[self.idf.income_column].values
        pline = _get_pline(y, pline, factor, q)
        n = len(y)
        ys = np.sort(y)
        q = np.sum(ys < pline)
        ii = np.arange(q)
        u = np.sum(np.dot(n - ii + 1, pline - ys[:q]))
        return (2 / (n * (n + 1) * pline)) * u

    def bd(self, pline=None, alpha=2, factor=1.0, q=None):
        """Blackorby and Donaldson Indices.

        Blackorby y Donaldson (1980) proponen una medida de pobreza de tipo
        normativo.

        Parameters
        ----------
        pline : optional(default=None)
            Absolute poverty line if pline is float.
            Relative poverty line if pline is 'median', 'quantile' or 'mean'
            If pline is None then pline = 0.5*median(y).

        factor : float, optional(default=1.0)
            Factor in pline = factor*stat

        q : float, optional(default=None)
            Cuantil q if pline is'quantile'

        alpha: float, optional(default=2)
            Aversion parameter. (ver)

        Return
        ------
        out: float
            Index measure.

        References
        ----------
        .. Blackorby, C. y Donaldson, D. (1980). “Ethical indices for the
           measurement of poverty”. Econometrica. Vol. 48, n 4,
           pp.1053–1060.

        """
        y = self.idf.data[self.idf.income_column].values
        pline = _get_pline(y, pline, factor, q)
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

    def hagenaars(self, pline=None, factor=1.0, q=None):
        """Hagenaars Index.

        Hagenaars (1984) para obtener la medida de pobreza considera
        la función de evaluación social de la renta como V(x) = ln(x).

        Parameters
        ----------
        pline : optional(default=None)
            Absolute poverty line if pline is float.
            Relative poverty line if pline is 'median', 'quantile' or 'mean'
            If pline is None then pline = 0.5*median(y).

        factor : float, optional(default=1.0)
            Factor in pline = factor*stat

        q : float, optional(default=None)
            Cuantil q if pline is'quantile'

        Return
        ------
        out: float
            Index measure.

        References
        ----------
        .. Hagenaars, A. (1984). “A class of poverty indices”. Center
           for Research in Public Economics. Leyden University.

        """
        y = self.idf.data[self.idf.income_column].values
        pline = _get_pline(y, pline, factor, q)
        n = len(y)
        ys = np.sort(y)
        q = np.sum(ys < pline)
        if q == 0:
            return 0  # check this!!
        yp = ys[0:q]
        ug = np.exp(sum(np.log(yp)) / q)  # o normalizar con el maximo
        return (q / n) * ((np.log(pline) - np.log(ug)) / np.log(pline))

    def chakravarty(self, pline=None, alpha=0.5, factor=1.0, q=None):
        """Chakravarty Indices.

        Chakravarty (1983) es una medida ética de pobreza. El índice de pobreza
        se obtiene como la suma normalizada de las carencias de
        utilidad de los pobres.

        Parameters
        ----------
        pline : optional(default=None)
            Absolute poverty line if pline is float.
            Relative poverty line if pline is 'median', 'quantile' or 'mean'
            If pline is None then pline = 0.5*median(y).

        factor : float, optional(default=1.0)
            Factor in pline = factor*stat

        q : float, optional(default=None)
            Cuantil q if pline is'quantile'

        alpha: float, optional(default=0.5)
            Aversion parameter. (ver)

        Return
        ------
        out: float
            Index measures.

        References
        ----------
        .. Chakravarty, S.R. (1983). “A new index of poverty”. Mathematical
           Social Sciences. Vol. 6, pp.307–313.

        """
        y = self.idf.data[self.idf.income_column].values
        pline = _get_pline(y, pline, factor, q)
        if (alpha <= 0) or (alpha >= 1):
            raise ValueError(f"'alpha' must be in (0,1). Found '{alpha}'")
        n = len(y)
        ys = np.sort(y)
        q = np.sum(ys < pline)
        yp = ys[0:q]
        return sum(1 - np.power(yp / pline, alpha)) / n


def _get_pline(y, pline, factor, q):
    """Check/calcule poverty line."""
    if pline is None:
        return 0.5 * np.median(y)
    if factor < 0:
        raise ValueError(f"'factor' must be <=0. Found '{factor}'")
    if pline == "median":
        return factor * np.median(y)
    elif pline == "mean":
        return factor * np.mean(y)
    elif pline == "quantile":
        if (q < 0) or (q > 1):
            raise ValueError(f"Quantile 'q' must be in [0,1]. Found '{q}'")
        return factor * np.quantile(y, q=q)
    elif pline <= 0:
        raise ValueError(f"'pline' must be >= 0. Found '{pline}'")
    else:
        return pline
