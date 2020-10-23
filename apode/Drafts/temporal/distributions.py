# Generacion de distribuciones
import numpy as np
import pandas as pd


def distribution_examples(rg, dist, n, nbin=None):
    y = distribution_examples_aux(rg, dist, n)
    df = pd.DataFrame({"x": y})
    if nbin is None:
        return df
    else:
        return binning(df, nbin=nbin)


def distribution_examples_aux(rg, dist, n):
    if dist == "uniform":
        mu = 100
        y = rg.uniform(size=n) * mu
    elif dist == "lognormal":
        y = rg.lognormal(mean=3.3, sigma=1.0, size=n)
    elif dist == "exponential":
        y = 50 * rg.exponential(scale=1, size=n)
    elif dist == "pareto":
        y = 200 * rg.pareto(a=5, size=n)
    elif dist == "chisquare":
        y = 10 * rg.chisquare(df=5, size=n)
    elif dist == "gamma":
        shape, scale = 1, 50.0  # mean=4, std=2*sqrt(2)
        y = rg.gamma(shape, scale, size=n)
    elif dist == "weibull":
        a = 1.5  # shape
        y = 50 * rg.weibull(a, size=n)
    return y


# generalizar columnanme
# puede remover nan
def binning(df, pos=0, nbin=None):
    if nbin is None:
        nbin = int(np.sqrt(df.shape[0]))
    s1 = df.groupby(pd.cut(df.iloc[:, pos], nbin)).count()
    s2 = df.groupby(pd.cut(df.iloc[:, pos], nbin)).mean()
    dfb = pd.concat([s1, s2], axis=1).dropna()
    dfb.columns = ["weight", "x"]
    dfb.reset_index(drop=True, inplace=True)
    return dfb
