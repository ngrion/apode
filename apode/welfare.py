# ----------------------------  Welfare -------------------------------
import numpy as np
#from .inequality import gini_s, entropy
 # se podria agregar todos los de desigualdad entre 0 y 1

 
import attr



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

    def isoelastic(self, alpha):
        y = self.idf.data[self.idf.varx].values
        if alpha == 0:
            return np.mean(y)
        elif alpha == np.Inf:
            return np.min(y)
        elif alpha == 1:
            return (1 / len(y)) * np.sum(np.log(y))
        return (1 / len(y)) * np.sum(np.power(y, 1 - alpha)) / (1 - alpha)

    def sen(self):
        y = self.idf.data[self.idf.varx].values
        u = np.mean(y)
        g = self.idf.inequality.gini(sort=True)
        return u * (1 - g)

    def theill(self):
        y = self.idf.data[self.idf.varx].values
        u = np.mean(y)
        tl = self.idf.inequality.entropy(alpha=0, sort=True)
        return u * np.exp(-tl)

    def theilt(self):
        y = self.idf.data[self.idf.varx].values
        u = np.mean(y)
        tt = self.idf.inequality.entropy(alpha=1, sort=True)
        return u * np.exp(-tt)






   

