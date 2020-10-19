import numpy as np
from .inequality import gini_s


def concentration_measure(y, method, *args):
    if method == "hhi":
        c = hhi(y, normalized=False)
    elif method == "hhin":
        c = hhi(y, normalized=True)
    elif method == "rosenbluth":
        c = rosenbluth(y)
    elif method == "cr":
        k = args[0]
        c = cr(y, k)
    else:
        raise ValueError("Método " + method + " no implementado.")
    return c


def concentration_measure_w(y, w, method, *args):
    raise ValueError("Método " + method +
                     " no implementado (datos agrupados).")
    c = []
    return c


# Herfindahl-Hirschman index
def hhi(y, normalized=True):
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
def rosenbluth(y):
    n = len(y)
    g = gini_s(y)
    return 1 / (n * (1 - g))


#  Concentration Ratio
def cr(y, k):
    n = len(y)
    if k < 0 or k > n:
        raise TypeError("n must be an positive integer "
                        "smaller than the data size")
    else:
        ys = np.sort(y)[::-1]
        return ys[:k].sum() / ys.sum()
