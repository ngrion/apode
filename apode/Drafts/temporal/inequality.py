# ----------------------------  Inequality -------------------------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ver ordenamiento!
# ver si vale controlar la cantidad exacat de argumentos
def ineq_measure(y, method, *args):
    if method in ["rr", "dmr", "cv", "dslog"]:
        q = ineq_basic(y, method)
    elif method in ["gini", "merhan", "piesch"]:
        q = ineq_linear(y, method)
    elif method == "bonferroni":
        q = bonferroni(y)
    elif method == "kolm":
        a = args[0]
        q = kolm(y, a)  # a > 0
    elif method == "ratio":
        r = args[0]
        q = ratio(y, r)  # 0 < r < 1
    elif method == "entropy":
        a = args[0]
        q = entropy(y, a)
    elif method == "atkinson":
        a = args[0]
        q = atkinson(y, a)
    else:
        raise ValueError("Método " + method + " no implementado.")
    return q


# Supone ordenados
def ineq_measure_w(ys, w, method, *args):
    # n = len(ys)
    u = sum(ys * w) / sum(w)
    # Relative range
    if method == "rr":
        q = (max(ys) - min(ys)) / u
    else:
        raise ValueError("Método " + method + " no implementado" " (datos agrupados).")
    return q


def ineq_basic(y, c):
    n = len(y)
    u = np.mean(y)
    # Relative range
    if c == "rr":
        q = (max(y) - min(y)) / u
    # Desviacion media relativa -Pietra o Schutz
    elif c == "dmr":
        q = sum(abs(y - u)) / (2 * n * u)
        # Coeficiente de variacion
    elif c == "cv":
        q = np.std(y) / u
    # Desv Est de los logaritmos
    elif c == "dslog":
        q = np.sqrt(sum(pow((np.log(u) - np.log(y)), 2)) / n)
    return q


def ineq_linear(y, c):
    n = len(y)
    u = np.mean(y)
    f = 1.0 / (n * u)
    syi = y[0]
    pi = 1.0 / n
    qi = f * y[0]
    m = pi - qi

    if c == "gini":  # no da igual
        for i in range(1, n - 1):  # =2,n-1
            pi = i / n
            syi = syi + y[i]
            qi = f * syi
            m = m + (pi - qi)
        m = m * 2 / n
    elif c == "merhan":
        for i in range(1, n - 1):  # do i=2,n-1
            pi = i / n
            syi = syi + y[i]
            qi = f * syi
            m = m + (1 - pi) * (pi - qi)
        m = m * 6 / n
    elif c == "piesch":
        for i in range(1, n - 1):  # 2,n-1
            pi = i / n
            syi = syi + y[i]
            qi = f * syi
            m = m + pi * (pi - qi)
        m = m * 3 / n
    return m


def bonferroni(y):
    n = len(y)
    s = y[0]
    x = y[0]
    for i in range(1, n - 1):  # i=2,n-1
        x = x + y[i]
        s = s + x / i
    u = (x + y[n - 1]) / n
    r = 1 - (1 / ((n - 1) * u)) * s
    return r


def kolm(y, a):  # a > 0
    n = len(y)
    u = np.mean(y)
    r = (1 / a) * (np.log((1.0 / n) * sum(np.exp(a * (u - y)))))
    return r


# Ratios de Kuznets (toma limite inferior)
def ratio(y, r):  # 0 < r < 1
    n = len(y)
    k = int(np.floor(r * n))
    q = np.mean(y[:k]) / np.mean(y[n - k :])
    return q


# tipo = 'r' o None  simple
# tipo = 'g'  generalizada
#  tipo = 'a' absoluta
def lorenz_curve(y, type=None, plot=True):
    n = len(y)

    z = np.cumsum(y) / y.sum()
    q = np.arange(0, n + 1) / n
    qd = q
    if (type == "r") or (type == "None"):
        pass
    elif type == "g":
        mu = np.mean(y)
        z = z * mu
        qd = q * mu
    elif type == "a":
        mu = np.mean(y)
        qd = q * 0
        z = np.cumsum(y - mu)

    z = np.insert(z, 0, 0)

    df = pd.DataFrame({"population": q, "variable": z})

    if plot:
        plt.plot(q, z)
        plt.plot(q, qd)
        plt.xlabel("Cumulative % of population")
        if (type == "r") or (type == "None"):
            plt.ylabel("Cumulative % of variable")
            plt.title("Lorenz Curve")
        elif type == "g":
            plt.ylabel("Scaled Cumulative % of variable")
            plt.title("Generalized Lorenz Curve")
        elif type == "a":
            plt.ylabel("Cumulative deviaton")
            plt.title("Absolut Lorenz Curve")
        plt.show()

    return df


def pen_parade(y, plot=True, pline=None):
    n = len(y)
    me = np.median(y)
    q = np.arange(0, n + 1) / n
    mu = np.mean(y)
    qd = np.ones(n + 1) * mu / me
    z = np.copy(y) / me
    z = np.insert(z, 0, 0)
    df = pd.DataFrame({"population": q, "variable": z})
    if plot:
        plt.plot(q, z)
        plt.plot(q, qd, label="Mean")
        if not (pline is None):
            qpl = np.ones(n + 1) * pline / me
            plt.plot(q, qpl, label="Poverty line")
        plt.xlabel("Cumulative % of population")
        plt.ylabel("Medianized variable")
        plt.title("Pen's Parade")
        plt.legend()
        plt.show()
    return df


# Entropia general
def entropy(y, a):
    n = len(y)
    u = np.mean(y)

    if a == 0.0:
        p = sum(np.log(u / y)) / n
    elif a == 1.0:
        p = sum((y / u) * np.log(y / u)) / n
    else:
        p = (1 / (a * (a - 1))) * (sum(pow(y / u, a)) / n - 1)
    return p


# invocados por pobreza
def gini_s(y):
    n = len(y)
    u = np.mean(y)
    a = 0
    for i in range(0, n):
        a = a + (n - i + 1) * y[i]
    g = (n + 1) / (n - 1) - 2 / (n * (n - 1) * u) * a
    g = g * (n - 1) / n
    return g


# Calculate the Atkinson inequality index.
def atkinson(y, epsilon):
    n = len(y)
    if n == 0:
        return 0
    else:
        if epsilon <= 0:
            raise TypeError("Epsilon must be strictly positive (>0.0)")
        elif epsilon == 1:
            y_nz = y[y != 0]
            ylog = np.log(y_nz)
            h = np.mean(ylog)
            return 1 - n * np.exp(h)
        else:
            n2 = np.power(n, epsilon / (epsilon - 1.0))
            h1 = np.power(y, 1.0 - epsilon).sum()
            h2 = np.power(h1, 1.0 / (1.0 - epsilon))
            return 1 - n2 * h2
