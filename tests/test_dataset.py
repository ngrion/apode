#!/usr/bin/env python
# -*- coding: utf-8 -*-

# This file is part of the
#   Apode Project (https://github.com/mchalela/apode).
# Copyright (c) 2020, Néstor Grión and Sofía Sappia
# License: MIT
#   Full Text: https://github.com/ngrion/apode/blob/master/LICENSE.txt

from apode import datasets

import numpy as np


def test_pareto():
    expected = np.array(
        [
            19.67978966,
            165.16220522,
            60.25619344,
            40.0640775,
            6.9013978,
            6.90021521,
            2.40793077,
            99.035872,
            40.35903803,
            55.84387084,
        ]
    )
    expected_2bin = np.array([29.05156416, 132.09903861])
    data = datasets.make_pareto(seed=42, a=5, size=10, c=200, nbin=None)
    data_2bin = datasets.make_pareto(seed=42, a=5, size=10, c=200, nbin=2)
    np.testing.assert_array_almost_equal(data.data.x.values, expected, 6)
    np.testing.assert_array_almost_equal(
        data_2bin.data.x.values, expected_2bin, 6
    )


def test_uniform():
    expected = np.array(
        [
            37.45401188,
            95.07143064,
            73.19939418,
            59.86584842,
            15.60186404,
            15.59945203,
            5.80836122,
            86.61761458,
            60.11150117,
            70.80725778,
        ]
    )
    expected_2bin = np.array([18.61592229, 74.27884113])
    data = datasets.make_uniform(seed=42, size=10, mu=100, nbin=None)
    data_2bin = datasets.make_uniform(seed=42, size=10, mu=100, nbin=2)
    np.testing.assert_array_almost_equal(data.data.x.values, expected, 6)
    np.testing.assert_array_almost_equal(
        data_2bin.data.x.values, expected_2bin, 6
    )


def test_lognormal():
    expected = np.array(
        [
            44.55454429,
            23.61154291,
            51.81545885,
            124.34125678,
            21.45261663,
            21.45296884,
            131.52708704,
            58.40694063,
            16.95436994,
            46.64473427,
        ]
    )
    expected_2bin = np.array([35.61164704, 127.93417191])
    data = datasets.make_lognormal(seed=42, size=10, sigma=1.0, nbin=None)
    data_2bin = datasets.make_lognormal(seed=42, size=10, sigma=1.0, nbin=2)
    np.testing.assert_array_almost_equal(data.data.x.values, expected, 6)
    np.testing.assert_array_almost_equal(
        data_2bin.data.x.values, expected_2bin, 6
    )


def test_chisquare():
    expected = np.array(
        [
            59.66270731,
            39.38905915,
            36.79910273,
            36.79953616,
            108.43213482,
            70.07982828,
            30.9296841,
            61.34871817,
            50.85394342,
            7.88759493,
        ]
    )
    expected_2bin = np.array([33.77648675, 74.88084715])
    data = datasets.make_chisquare(seed=42, size=10, df=5, c=10, nbin=None)
    data_2bin = datasets.make_chisquare(seed=42, size=10, df=5, c=10, nbin=2)
    np.testing.assert_array_almost_equal(data.data.x.values, expected, 6)
    np.testing.assert_array_almost_equal(
        data_2bin.data.x.values, expected_2bin, 6
    )


def test_gamma():
    expected = np.array(
        [
            23.4634045,
            150.50607155,
            65.83728468,
            45.64712769,
            8.48124352,
            8.4798146,
            2.99193843,
            100.56154322,
            45.95410768,
            61.56250309,
        ]
    )
    expected_2bin = np.array([32.80217802, 125.53380738])
    data = datasets.make_gamma(
        seed=42, size=10, shape=1, scale=50.0, nbin=None
    )
    data_2bin = datasets.make_gamma(
        seed=42, size=10, shape=1, scale=50.0, nbin=2
    )
    np.testing.assert_array_almost_equal(data.data.x.values, expected, 6)
    np.testing.assert_array_almost_equal(
        data_2bin.data.x.values, expected_2bin, 6
    )


def test_weibull():
    expected = np.array(
        [
            30.19377075,
            104.23798675,
            60.06727664,
            47.05426256,
            15.32132613,
            15.31960518,
            7.64936004,
            79.66690664,
            47.26498885,
            57.43820348,
        ]
    )
    expected_2bin = np.array([27.13388559, 75.35259338])
    data = datasets.make_weibull(seed=42, size=10, a=1.5, c=50, nbin=None)
    data_2bin = datasets.make_weibull(seed=42, size=10, a=1.5, c=50, nbin=2)
    np.testing.assert_array_almost_equal(data.data.x.values, expected, 6)
    np.testing.assert_array_almost_equal(
        data_2bin.data.x.values, expected_2bin, 6
    )


def test_exponential():
    expected = np.array(
        [
            23.4634045,
            150.50607155,
            65.83728468,
            45.64712769,
            8.48124352,
            8.4798146,
            2.99193843,
            100.56154322,
            45.95410768,
            61.56250309,
        ]
    )
    expected_2bin = np.array([32.80217802, 125.53380738])
    data = datasets.make_exponential(
        seed=42, size=10, scale=1, c=50, nbin=None
    )
    data_2bin = datasets.make_exponential(
        seed=42, size=10, scale=1, c=50, nbin=2
    )
    np.testing.assert_array_almost_equal(data.data.x.values, expected, 6)
    np.testing.assert_array_almost_equal(
        data_2bin.data.x.values, expected_2bin, 6
    )


def test_constant():
    expected = np.array(
        [10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0]
    )
    expected_2bin = np.array([10.0])
    data = datasets.make_constant(size=10, nbin=None)
    data_2bin = datasets.make_constant(size=10, nbin=2)
    np.testing.assert_array_equal(data.data.x.values, expected)
    np.testing.assert_array_equal(data_2bin.data.x.values, expected_2bin)


def test_linear():
    expected = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    expected_2bin = np.array([3, 8])
    data = datasets.make_linear(size=10, nbin=None)
    data_2bin = datasets.make_linear(size=10, nbin=2)
    np.testing.assert_array_equal(data.data.x.values, expected)
    np.testing.assert_array_equal(data_2bin.data.x.values, expected_2bin)


def test_squared():
    expected = np.array([1, 4, 9, 16, 25, 36, 49, 64, 81, 100])
    expected_2bin = np.array([20.0, 81.66666667])
    data = datasets.make_squared(size=10, nbin=None)
    data_2bin = datasets.make_squared(size=10, nbin=2)
    np.testing.assert_array_equal(data.data.x.values, expected)
    np.testing.assert_array_almost_equal(
        data_2bin.data.x.values, expected_2bin, 6
    )


def test_extreme():
    expected = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 10.0])
    expected_2bin = np.array([0.0, 10.0])
    data = datasets.make_extreme(size=10, nbin=None)
    data_2bin = datasets.make_extreme(size=10, nbin=2)
    np.testing.assert_array_equal(data.data.x.values, expected)
    np.testing.assert_array_equal(data_2bin.data.x.values, expected_2bin)


def test_unimodal():
    expected = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 10.0])
    expected_2bin = np.array([0.0, 10.0])
    data = datasets.make_unimodal(size=10, nbin=None)
    data_2bin = datasets.make_unimodal(size=10, nbin=2)
    np.testing.assert_array_equal(data.data.x.values, expected)
    np.testing.assert_array_equal(data_2bin.data.x.values, expected_2bin)


def test_bimodal():
    expected = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 10.0])
    expected_2bin = np.array([0.0, 10.0])
    data = datasets.make_bimodal(size=10, nbin=None)
    data_2bin = datasets.make_bimodal(size=10, nbin=2)
    np.testing.assert_array_equal(data.data.x.values, expected)
    np.testing.assert_array_equal(data_2bin.data.x.values, expected_2bin)
