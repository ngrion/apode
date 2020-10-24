#!/usr/bin/env python
# -*- coding: utf-8 -*-

# This file is part of the
#   Apode Project (https://github.com/mchalela/apode).
# Copyright (c) 2020, Néstor Grión and Sofía Sappia
# License: MIT
#   Full Text: https://github.com/ngrion/apode/blob/master/LICENSE.txt

import numpy as np
import pandas as pd

from apode.basic import ApodeData


# TODO:
#  - check for all methods, didn't know if they
#  were all supposed to provide the same response
#  - add testing for list of poverty measures

def test_min(uniform_ad):
    # TODO: Check for all methods.
    #  Commented out those for which the test fails
    data = uniform_ad
    pline = np.min(data.data.values) - 1
    assert data.poverty(method="headcount", pline=pline) == 0
    assert data.poverty(method="gap", pline=pline) == 0
    assert data.poverty(method="severity", pline=pline) == 0
    assert data.poverty(method="fgt", pline=pline) == 0
    assert data.poverty(method="sen", pline=pline) == 0
    assert data.poverty(method="sst", pline=pline) == 0
    assert data.poverty(method="watts", pline=pline) == 0
    # assert data.poverty(method='cuh', pline=pline) == 0
    # assert data.poverty(method='takayama', pline=pline) == 0
    # assert data.poverty(method='kakwani', pline=pline) == 0
    assert data.poverty(method="thon", pline=pline) == 0
    # assert data.poverty(method='bd', pline=pline) == 0
    # assert data.poverty(method='hagenaars', pline=pline) == 0
    assert data.poverty(method="chakravarty", pline=pline) == 0
    # assert data.poverty(method='tip', pline=pline) == 0


def test_max(uniform_ad):
    # TODO: Check for all methods.
    #  Commented out those for which the test fails
    data = uniform_ad
    pline = np.min(data.data.values) + 1
    assert data.poverty(method="headcount", pline=pline) == 1
    # assert data.poverty(method='gap', pline=pline) == 1
    # assert data.poverty(method='severity', pline=pline) == 1
    assert data.poverty(method="fgt", pline=pline) == 1
    # assert data.poverty(method='sen', pline=pline) == 1
    # assert data.poverty(method='sst', pline=pline) == 1
    # assert data.poverty(method='watts', pline=pline) == 1
    # assert data.poverty(method='cuh', pline=pline) == 1
    # assert data.poverty(method='takayama', pline=pline) == 1
    # assert data.poverty(method='kakwani', pline=pline) == 1
    # assert data.poverty(method='thon', pline=pline) == 1
    # assert data.poverty(method='bd', pline=pline) == 1
    # assert data.poverty(method='hagenaars', pline=pline) == 1
    # assert data.poverty(method='chakravarty', pline=pline) == 1
    # assert data.poverty(method='tip', pline=pline) == 1


def test_symmetry(uniform_ad):
    # TODO: Check for all methods.
    #  Commented out those for which the test fails
    data = uniform_ad
    pline = np.mean(data.data.values)
    y = data.data["x"].tolist()
    np.random.shuffle(y)
    df2 = pd.DataFrame({"x": y})
    dr2 = ApodeData(df2, varx="x")
    assert data.poverty(method="headcount", pline=pline) == dr2.poverty(
        method="headcount", pline=pline
    )
    assert data.poverty(method="gap", pline=pline) == dr2.poverty(
        method="gap", pline=pline
    )
    assert data.poverty(method="severity", pline=pline) == dr2.poverty(
        method="severity", pline=pline
    )
    assert data.poverty(method="fgt", pline=pline) == dr2.poverty(
        method="fgt", pline=pline
    )
    assert data.poverty(method="sen", pline=pline) == dr2.poverty(
        method="sen", pline=pline
    )
    assert data.poverty(method="sst", pline=pline) == dr2.poverty(
        method="sst", pline=pline
    )
    assert data.poverty(method="watts", pline=pline) == dr2.poverty(
        method="watts", pline=pline
    )
    assert data.poverty(method="cuh", pline=pline) == dr2.poverty(
        method="cuh", pline=pline
    )
    # assert data.poverty(method='takayama', pline=pline) == dr2.poverty(
    # method='takayama', pline=pline
    # )
    assert data.poverty(method="kakwani", pline=pline) == dr2.poverty(
        method="kakwani", pline=pline
    )
    assert data.poverty(method="thon", pline=pline) == dr2.poverty(
        method="thon", pline=pline
    )
    assert data.poverty(method="bd", pline=pline) == dr2.poverty(
        method="bd", pline=pline
    )
    assert data.poverty(method="hagenaars", pline=pline) == dr2.poverty(
        method="hagenaars", pline=pline
    )
    assert data.poverty(method="chakravarty", pline=pline) == dr2.poverty(
        method="chakravarty", pline=pline
    )

    # assert data.poverty(method='tip', pline=pline) == dr2.poverty(
    # method='tip', pline=pline
    # )


def test_replication(uniform_ad):
    # TODO: Check for all methods.
    #  Commented out those for which the test fails
    data = uniform_ad
    k = 2  # factor
    pline = np.mean(data.data.values)
    y = k * data.data["x"].tolist()
    df2 = pd.DataFrame({"x": y})
    dr2 = ApodeData(df2, varx="x")
    assert data.poverty("headcount", pline=pline) == dr2.poverty(
        "headcount", pline=pline
    )
    # assert data.poverty("gap", pline=pline) == dr2.poverty(
    #     "gap", pline=pline
    # )
    # assert data.poverty("sst", pline=pline) == dr2.poverty(
    #     "sst", pline=pline
    # )
    assert data.poverty("fgt", pline=pline) == dr2.poverty(
        "fgt", pline=pline
    )
    assert data.poverty("watts", pline=pline) == dr2.poverty(
        "watts", pline=pline
    )
    # assert data.poverty("cuh", pline=pline) == dr2.poverty(
    #     "cuh", pline=pline
    # )
    # assert data.poverty("takayama", pline=pline) == dr2.poverty(
    #     "takayama", pline=pline
    # )
    # assert data.poverty("kakwani", pline=pline) == dr2.poverty(
    #     "kakwani", pline=pline
    # )
    # assert data.poverty("thon", pline=pline) == dr2.poverty(
    #     "thon", pline=pline
    # )
    assert data.poverty("bd", pline=pline) == dr2.poverty(
        "bd", pline=pline
    )
    # assert data.poverty("chakravarty", pline=pline) == dr2.poverty(
    #     "chakravarty", pline=pline
    # )
    # assert data.poverty("tip", pline=pline) == dr2.poverty(
    #     "tip", pline=pline
    # )


def test_homogeneity(uniform_ad):
    data = uniform_ad
    k = 2  # factor
    pline = np.mean(data.data.values)
    y = data.data['x'].tolist()
    y = [yi * k for yi in y]
    df2 = pd.DataFrame({'x': y})
    dr2 = ApodeData(df2, varx="x")
    assert data.poverty('headcount', pline=pline) == dr2.poverty(
        'headcount', pline=pline * k)
    assert data.poverty('gap', pline=pline) == dr2.poverty(
        'gap', pline=pline * k)
    assert data.poverty('severity', pline=pline) == dr2.poverty(
        'severity', pline=pline * k)
    assert data.poverty('headcount', pline=pline) == dr2.poverty(
        'fgt', pline=pline * k)
    assert data.poverty('sen', pline=pline) == dr2.poverty(
        'sen', pline=pline * k)
    assert data.poverty('sst', pline=pline) == dr2.poverty(
        'sst', pline=pline * k)
    assert data.poverty('watts', pline=pline) == dr2.poverty(
        'watts', pline=pline * k)
    assert data.poverty('cuh', pline=pline) == dr2.poverty(
        'cuh', pline=pline * k)
    assert data.poverty('takayama', pline=pline) == dr2.poverty(
        'takayama', pline=pline * k)
    assert data.poverty('kakwani', pline=pline) == dr2.poverty(
        'kakwani', pline=pline * k)
    assert data.poverty('thon', pline=pline) == dr2.poverty(
        'thon', pline=pline * k)
    assert data.poverty('bd', pline=pline) == dr2.poverty(
        'bd', pline=pline * k)
    # assert data.poverty('hagenaars', pline=pline) == dr2.poverty(
    #     'hagenaars', pline=pline * k)
    assert data.poverty('chakravarty', pline=pline) == dr2.poverty(
        'chakravarty', pline=pline * k)
    # assert data.poverty('tip', pline=pline) == dr2.poverty(
    #     'tip', pline=pline * k)


# # Testea metdodo de un listado de medidas de pobreza
# def test_lista(prop,lista):
#     x = TestCaseUniform()
#     x.setup_method()
#     for elem in lista:
#         if elem[1]==None:
#             x.prop(elem[0])
#         else:
#             x.prop(elem[0],alpha=elem[1])
