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

    def gini(self):
        y = self.idf.data[self.idf.varx].values
        n = len(y)
        u = np.mean(y)
        a = 0
        for i in range(0, n):
            a = a + (n - i + 1) * y[i]
        g = (n + 1) / (n - 1) - 2 / (n * (n - 1) * u) * a
        g = g * (n - 1) / n
        return g


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
class ApodeData:
    data = attr.ib(converter=pd.DataFrame)
    varx = attr.ib()
    poverty = attr.ib(init=False)
    inequity = attr.ib(init=False)
    polarization = attr.ib(init=False)

    @poverty.default
    def _poverty_default(self):
        return PovertyMeasures(idf=self)

    @inequity.default
    def _inequity_default(self):
        return InequityMeasures(idf=self)

    @polarization.default
    def _polarization_default(self):
        return PolarizationMeasures(idf=self)

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
