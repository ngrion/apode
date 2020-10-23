#!/usr/bin/env python
# -*- coding: utf-8 -*-

# This file is part of the
#   Apode Project (https://github.com/mchalela/apode).
# Copyright (c) 2020, Néstor Grión and Sofía Sappia
# License: MIT
#   Full Text: https://github.com/ngrion/apode/blob/master/LICENSE.txt


import pytest
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from apode.basic import ApodeData
from apode.poverty import PovertyMeasures 

from numpy.testing import assert_equal, assert_, assert_almost_equal

