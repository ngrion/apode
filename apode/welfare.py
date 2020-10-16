# ----------------------------  Welfare -------------------------------
import numpy as np
from .inequality import gini_s, entropy


# se podria agregar todos los de desigualdad entre 0 y 1
# pero debería subirse un nivel (idem pobreza)
def welfare_measure(y, method, *args):
    if method == "utilitarian":
        w = np.mean(y)
    elif method == "rawlsian":
        w = min(y)
    elif method == "isoelastic":
        e = args[0]
        if e == 0:
            w = np.mean(y)
        elif e == np.Inf:
            w = min(y)
        elif e == 1:
            w = (1 / len(y)) * sum(np.log(y))
        else:
            w = (1 / len(y)) * sum(np.power(y, 1 - e)) / (1 - e)
    elif method == "sen":
        u = np.mean(y)
        g = gini_s(np.sort(y))
        w = u * (1 - g)
    elif method == "theill":
        u = np.mean(y)
        tl = entropy(np.sort(y), 0)
        w = u * np.exp(-tl)
    elif method == "theilt":
        u = np.mean(y)
        tt = entropy(np.sort(y), 1)
        w = u * np.exp(-tt)
    else:
        raise ValueError("Método " + method + " no implementado.")
    return w

    # se podria agregar todos los de desigualdad entre 0 y 1


# pero debería subirse un nivel (idem pobreza)
def welfare_measure_w(ys, w, method, *args):
    if method == "utilitarian":
        w = sum(ys * w) / sum(w)
    elif method == "rawlsian":
        w = min(ys)
    elif method == "isoelastic":
        e = args[0]
        if e == 0:
            w = sum(ys * w) / sum(w)
        elif e == np.Inf:
            w = min(ys)
        elif e == 1:
            n = sum(w)
            w = (1 / n) * sum(w * np.log(ys))
        else:
            w = (1 / n) * sum(w * np.power(ys, 1 - e)) / (1 - e)
    else:
        raise ValueError("Método " + method + " no implementado "
                                              "(datos agrupados).")
    return w
