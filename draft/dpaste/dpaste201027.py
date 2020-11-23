import pytest
import numpy as np
import pandas as pd

from apode.basic import ApodeData


@pytest.fixture(scope="session")
def uniform_ad():
    def make(*, seed, **kwargs):
        random = np.random.RandomState(seed=seed)
        x = random.uniform(**kwargs)
        df1 = pd.DataFrame({"x": x})
        return ApodeData(df1, varx="x")

    return make


@pytest.fixture(scope="session")
def normal_ad():
    def make(*, seed, **kwargs):
        random = np.random.RandomState(seed=seed)
        x = random.normal(**kwargs)
        df1 = pd.DataFrame({"x": x})
        return ApodeData(df1, varx="x")

    return make


# see USA asi
def test_somethinf(uniform_ad):
    data = uniform_ad(seed=42, size=10)
