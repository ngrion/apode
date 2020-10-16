
  1
  2
  3
  4
  5
  6
  7
  8
  9
 10
 11
 12
 13
 14
 15
 16
 17
 18
 19
 20
 21
 22
 23
 24
 25
 26
 27
 28
 29
 30
 31
 32
 33
 34
 35
 36
 37
 38
 39
 40
 41
 42
 43
 44
 45
 46
 47
 48
 49
 50
 51
 52
 53
 54
 55
 56
 57
 58
 59
 60
 61
 62
 63
 64
 65
 66
 67
 68
 69
 70
 71
 72
 73
 74
 75
 76
 77
 78
 79
 80
 81
 82
 83
 84
 85
 86
 87
 88
 89
 90
 91
 92
 93
 94
 95
 96
 97
 98
 99
100
101
102
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
class ApodeData:

    data = attr.ib(converter=pd.DataFrame)
    varx = attr.ib()
    poverty = attr.ib(init=False)
    inequity = attr.ib(init=False)

    @poverty.default
    def _poverty_default(self):
        return PovertyMeasures(idf=self)

    @inequity.default
    def _inequity_default(self):
        return InequityMeasures(idf=self)

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