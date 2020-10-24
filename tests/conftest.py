import pytest
import numpy as np
import pandas as pd

from apode.basic import ApodeData

random = np.random.RandomState(seed=42)


@pytest.fixture
def uniform_ad():
    x = random.uniform(size=10)
    df1 = pd.DataFrame({"x": x})
    return ApodeData(df1, varx="x")


@pytest.fixture
def normal_ad():
    x = random.normal(size=10)
    df1 = pd.DataFrame({"x": x})
    return ApodeData(df1, varx="x")
