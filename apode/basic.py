#import numpy as np
import pandas as pd

import attr

from .poverty import PovertyMeasures #tip_curve
from .inequality import InequalityMeasures # lorenz_curve, pen_parade
from .welfare import WelfareMeasures
from .polarization import PolarizationMeasures
from .concentration import ConcentrationMeasures


@attr.s(frozen=True)
class ApodeData:
    data = attr.ib(converter=pd.DataFrame)
    varx = attr.ib()
    poverty = attr.ib(init=False)
    inequality = attr.ib(init=False)
    polarization = attr.ib(init=False)
    concentration = attr.ib(init=False)
    welfare = attr.ib(init=False)

    @poverty.default
    def _poverty_default(self):
        return PovertyMeasures(idf=self)

    @inequality.default
    def _inequality_default(self):
        return InequalityMeasures(idf=self)

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
