import numpy as np
#from .inequality import gini_s

import attr


@attr.s(frozen=True)
class ConcentrationMeasures:
    idf = attr.ib()

    def __call__(self, method=None, **kwargs):
        method = "herfindahl" if method is None else method
        method_func = getattr(self, method)
        return method_func(**kwargs)

    # Herfindahl-Hirschman index
    def herfindahl(self, normalized=True):
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
        g = self.idf.inequality.gini()
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

