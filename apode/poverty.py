# ver valor de pline (no puede ser 1, por log)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#from .inequality import gini_s, atkinson

import attr

@attr.s(frozen=True)
class PovertyMeasures:
    idf = attr.ib()

    def __call__(self, method=None, **kwargs):
        method = "headcount" if method is None else method
        method_func = getattr(self, method)
        return method_func(**kwargs)

    # FGT(0)
    def headcount(self, pline):
        y = self.idf.data[self.idf.varx].values
        n = len(y)
        ys = np.sort(y)
        q = np.sum(ys < pline)
        return q / n

    # FGT(1)
    def gap(self, pline):
        y = self.idf.data[self.idf.varx].values
        n = len(y)
        ys = np.sort(y)
        q = np.sum(ys < pline)
        yp = ys[0:q]
        br = (pline - yp) / pline
        return np.sum(br) / n

    # FGT(2)
    def severity(self, pline):
        y = self.idf.data[self.idf.varx].values
        n = len(y)
        ys = np.sort(y)
        q = np.sum(ys < pline)
        yp = ys[0:q]
        br = np.power((pline - yp) / pline, 2)
        return np.sum(br) / n

    # FGT(alpha)  Foster–Greer–Thorbecke Index
    def fgt(self, pline, alpha=0):
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


    # Sen Index
    def sen(self, pline):
        p0 = self.headcount(pline=pline)
        p1 = self.gap(pline=pline)
        gp = self.idf.inequality.gini()
        return p0 * gp + p1 * (1 - gp)


    #  Sen-Shorrocks-Thon Index
    def sst(self, pline):
        p0 = self.headcount(pline=pline)
        p1 = self.gap(pline=pline)
        gp = self.idf.inequality.gini()
        return p0 * p1 * (1 + gp)


    # Watts index (1968)
    def watts(self, pline):
        y = self.idf.data[self.idf.varx].values
        n = len(y)
        ys = np.sort(y)
        q = np.sum(ys < pline)
        yp = ys[0:q]
        return sum(np.log(pline / yp)) / n


   # Indice de Clark, Ulph y Hemming (1981)
    def cuh(self, pline, alpha=0):
        y = self.idf.data[self.idf.varx].values
        n = len(y)
        ys = np.sort(y)
        q = np.sum(ys < pline)
        yp = ys[0:q]
        if (alpha < 0) or (alpha > 1) :
            raise ValueError(f"'alpha' must be in [0,1]. Found '{alpha}'")
        if alpha == 0:
            return 1 - np.power(np.product(yp / pline) / n, 1 / n)
        else:
            return 1 - np.power(
                (sum(np.power(yp / pline, alpha)) + (n - q)) / n, 1 / alpha
            )


    # Indice de Takayama
    def takayama(self, pline):
        y = self.idf.data[self.idf.varx].values
        n = len(y)
        ys = np.sort(y)
        q = np.sum(ys < pline)
        yp = ys[0:q]
        u = (yp.sum() + (n - q) * pline) / n
        a = 0
        for i in range(0, q):
            a = a + (n - i + 1) * y[i]
        for i in range(q, n):
            a = a + (n - i + 1) * pline
        return 1 + 1 / n - (2 / (u * n * n)) * a


    # Kakwani Index
    def kakwani(self, pline,  alpha=2):
        y = self.idf.data[self.idf.varx].values
        n = len(y)
        ys = np.sort(y)
        q = np.sum(ys < pline)
        yp = ys[0:q]
        #alpha = 2.0  # elegible
        a = 0.0
        u = 0.0
        for i in range(0, q):
            f = np.power(q - i + 2, alpha)  # ver +2
            a = a + f
            u = u + f * (pline - ys[i])
        return (q / (n * pline * a)) * u


    # Thon Index
    def thon(self, pline):
        y = self.idf.data[self.idf.varx].values
        n = len(y)
        ys = np.sort(y)
        q = np.sum(ys < pline)
        yp = ys[0:q]
        u = 0
        for i in range(0, q):
            u = u + (n - i + 1) * (pline - ys[i])
        return (2 / (n * (n + 1) * pline)) * u


    # Indice de Blackorby y Donaldson
    def bd(self, pline,  alpha=2):
        y = self.idf.data[self.idf.varx].values
        n = len(y)
        ys = np.sort(y)
        q = np.sum(ys < pline)
        yp = ys[0:q]    
        u = yp.sum() / q
        #atkp = atkinson(yp, alpha)
        #gp = self.idf.inequality.gini()
        atkp = self.idf.inequality.atkinson(alpha=alpha)
        yedep = u * (1 - atkp)
        return (q / n) * (pline - yedep) / pline


    # Hagenaars
    def hagenaars(self, pline):
        y = self.idf.data[self.idf.varx].values
        n = len(y)
        ys = np.sort(y)
        q = np.sum(ys < pline)
        yp = ys[0:q]        
        ug = np.exp(sum(np.log(yp)) / q)  # o normalizar con el maximo
        return (q / n) * ((np.log(pline) - np.log(ug)) / np.log(pline))


    # Chakravarty (1983)
    def chakravarty(self, pline, alpha=0.5):
        y = self.idf.data[self.idf.varx].values
        if (alpha <= 0) or (alpha >= 1) :
            raise ValueError(f"'alpha' must be in (0,1). Found '{alpha}'")
        n = len(y)
        ys = np.sort(y)
        q = np.sum(ys < pline)
        yp = ys[0:q]        
        return sum(1 - np.power(yp / pline, alpha)) / n


    # TIP Curve
    def tip(self, pline, plot=True):
        y = self.idf.data[self.idf.varx].values
        ys = np.sort(y)
        n = len(ys)
        q = sum(ys < pline)
        ygap = np.zeros(n)
        ygap[0:q] = (pline - ys[0:q]) / pline

        z = np.cumsum(ygap) / n
        z = np.insert(z, 0, 0)
        p = np.arange(0, n + 1) / n
        df = pd.DataFrame({"population": p, "variable": z})
        if plot:
            plt.plot(p, z)
            plt.title("TIP Curve")
            plt.ylabel("Cumulated poverty gaps")
            plt.xlabel("Cumulative % of population")
            plt.show()
        return df
