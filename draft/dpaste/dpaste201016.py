import numpy as np
import pandas as pd

import attr


@attr.s(frozen=True)
class PovertyMeasures:
    idf = attr.ib()

    def __call__(self, method=None, **kwargs):
        method = "foster" if method is None else method
        method_func = getattr(self, method)
        return method_func(**kwargs)

    def foster(self, pline, alpha=0):
        y = self.idf.data[self.idf.varx].values
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
        br = np.power((pline - yp) / pline, 2)
        return np.sum(br) / n

    def sen(self, pline):
        p0 = self.foster(alpha=0, pline=pline)
        p1 = self.foster(alpha=1, pline=pline)
        gp = self.idf.inequity.gini()
        return p0 * gp + p1 * (1 - gp)


@attr.s(frozen=True)
class InequityMeasures:
    idf = attr.ib()

    def __call__(self, method=None, **kwargs):
        method = "gini" if method is None else method
        method_func = getattr(self, method)
        return method_func(**kwargs)

    def gini(self, sort=False):
        y = self.idf.data[self.idf.varx].values
        if sort:
            y = np.sort(y)
        n = len(y)
        u = np.mean(y)
        a = 0
        for i in range(0, n):
            a = a + (n - i + 1) * y[i]
        g = (n + 1) / (n - 1) - 2 / (n * (n - 1) * u) * a
        g = g * (n - 1) / n
        return g

    # Entropia general
    def entropy(self, a=0, sort=False):
        y = self.idf.data[self.idf.varx].values
        if sort:
            y = np.sort(y)
        n = len(y)
        u = np.mean(y)
        if a == 0.0:
            return np.sum(np.log(u / y)) / n
        elif a == 1.0:
            return np.sum((y / u) * np.log(y / u)) / n
        return (1 / (a * (a - 1))) * (np.sum(pow(y / u, a)) / n - 1)


@attr.s(frozen=True)
class ConcentrationMeasures:
    idf = attr.ib()

    def __call__(self, method=None, **kwargs):
        method = "hhi" if method is None else method
        method_func = getattr(self, method)
        return method_func(**kwargs)

    # Herfindahl-Hirschman index
    def herfindahl_hirschman(self, normalized=True):
        y = self.idf.data[self.idf.varx].values
        w = y / sum(y)
        n = len(y)
        if n == 0:
            return 0
        else:
            h = np.square(w).sum()
            if normalized:
                return (h - 1.0 / n) / (1.0 - 1.0 / n)
            else:
                return h

    #  Rosenbluth index
    def rosenbluth(self):
        y = self.idf.data[self.idf.varx].values
        n = len(y)
        g = self.idf.inequity.gini()
        return 1 / (n * (1 - g))

    #  Concentration Ratio
    def concentration_ratio(self, k):
        y = self.idf.data[self.idf.varx].values
        n = len(y)
        if k < 0 or k > n:
            raise ValueError(
                "n must be an positive integer " "smaller than the data size"
            )
        else:
            ys = np.sort(y)[::-1]
            return ys[:k].sum() / ys.sum()


@attr.s(frozen=True)
class PolarizationMeasures:
    idf = attr.ib()

    def __call__(self, method=None, **kwargs):
        method = "esteban_ray" if method is None else method
        method_func = getattr(self, method)
        return method_func(**kwargs)

    # Esteban and Ray index of polarization
    # generalizar parametro
    def esteban_ray(self):
        y = self.idf.data[self.idf.varx].values
        n = len(y)
        alpha = 1  # (0,1.6]
        p_er = 0
        for i in range(len(y)):
            for j in range(len(y)):
                pi = 1 / n
                pj = 1 / n
                p_er += np.power(pi, 1 + alpha) * pj * abs(y[i] - y[j])
        return p_er

    # Wolfson index of bipolarization
    # ver que n> sea grande
    def wolfson_idx(self):
        ys = np.sort(self.idf.data[self.idf.varx].values)
        ysa = np.cumsum(ys) / np.sum(ys)
        n = len(ys)
        if (n % 2) == 0:
            i = int(n / 2)
            L = (ysa[i - 1] + ysa[i]) / 2
        else:
            i = int((n + 1) / 2)
            L = ysa[i - 1]
        g = self.idf.inequity.gini()
        p_w = (np.mean(ys) / np.median(ys)) * (0.5 - L - g)
        return p_w


@attr.s(frozen=True)
class WelfareMeasures:
    idf = attr.ib()

    def __call__(self, method=None, **kwargs):
        method = "utilitarian" if method is None else method
        method_func = getattr(self, method)
        return method_func(**kwargs)

    def utilitarian(self):
        y = self.idf.data[self.idf.varx].values
        return np.mean(y)

    def rawlsian(self):
        y = self.idf.data[self.idf.varx].values
        return min(y)

    def isoelastic(self, e):
        y = self.idf.data[self.idf.varx].values
        if e == 0:
            return np.mean(y)
        elif e == np.Inf:
            return np.min(y)
        elif e == 1:
            return (1 / len(y)) * np.sum(np.log(y))
        return (1 / len(y)) * np.sum(np.power(y, 1 - e)) / (1 - e)

    def sen(self):
        y = self.idf.data[self.idf.varx].values
        u = np.mean(y)
        g = self.idf.inequity.gini(sort=True)
        return u * (1 - g)

    def theill(self):
        y = self.idf.data[self.idf.varx].values
        u = np.mean(y)
        tl = self.idf.inequity.entropy(a=0, sort=True)
        return u * np.exp(-tl)

    def theilt(self):
        y = self.idf.data[self.idf.varx].values
        u = np.mean(y)
        tt = self.idf.inequity.entropy(a=1, sort=True)
        return u * np.exp(-tt)


@attr.s(frozen=True)
class ApodeData:
    data = attr.ib(converter=pd.DataFrame)
    varx = attr.ib()
    poverty = attr.ib(init=False)
    inequity = attr.ib(init=False)
    polarization = attr.ib(init=False)
    concentration = attr.ib(init=False)
    welfare = attr.ib(init=False)

    @poverty.default
    def _poverty_default(self):
        return PovertyMeasures(idf=self)

    @inequity.default
    def _inequity_default(self):
        return InequityMeasures(idf=self)

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

    # ~ def groupby(self, ...):
    # ~ return GroupedInequity(....)


# ~ class GroupedInequity(Inequity):


# ~ def groupby(self, ....):
# ~ raise NotImplementedError("GroupedInequity can't be grouped")


idf = ApodeData({"x": [23, 10, 12, 21, 4, 8, 19, 15, 11, 9]}, varx="x")
