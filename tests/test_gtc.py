"""
Compare with some of the linear fits of GTC.
"""
import math

import numpy as np
from GTC import get_correlation
from GTC import type_a
from pytest import approx

from msl.nlf import LinearModel
from msl.nlf import Model
from msl.nlf.datatypes import Result
from msl.nlf.parameter import InputParameters
from msl.nlf.parameter import ResultParameters


def test_line_fit():
    x = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    y = [15.6, 17.5, 36.6, 43.8, 58.2, 61.6, 64.2, 70.4, 98.8]

    gtc = type_a.line_fit(x, y)

    with LinearModel() as model:
        params = model.create_parameters([('a1', 1, False, 'intercept'),
                                          ('a2', 1, False, 'slope')])
        nlf = model.fit(x=x, y=y, params=params)

    assert approx(gtc.intercept.x) == nlf.params['intercept'].value
    assert approx(gtc.intercept.u) == nlf.params['intercept'].uncert
    assert approx(gtc.slope.x) == nlf.params['slope'].value
    assert approx(gtc.slope.u) == nlf.params['slope'].uncert
    assert approx(gtc.ssr) == nlf.chisq


def test_line_fit_wls():
    x = [1, 2, 3, 4, 5, 6]
    y = [3.2, 4.3, 7.6, 8.6, 11.7, 12.8]
    uy = [0.5, 0.5, 0.5, 1.0, 1.0, 1.0]

    gtc = type_a.line_fit_wls(x, y, uy)

    with LinearModel(weighted=True) as model:
        params = model.create_parameters([('a1', 1, False, 'intercept'),
                                          ('a2', 1, False, 'slope')])
        nlf = model.fit(x=x, y=y, params=params, uy=uy)

    assert approx(gtc.intercept.x) == nlf.params['intercept'].value
    assert approx(gtc.intercept.u) == nlf.params['intercept'].uncert
    assert approx(gtc.slope.x) == nlf.params['slope'].value
    assert approx(gtc.slope.u) == nlf.params['slope'].uncert
    assert approx(gtc.ssr) == nlf.chisq


def test_line_fit_wtls():
    # Pearson-York test data see, e.g.,
    # Lybanon, M. in Am. J. Phys 52 (1) 1984
    x = [0.0, 0.9, 1.8, 2.6, 3.3, 4.4, 5.2, 6.1, 6.5, 7.4]
    wx = [1000, 1000, 500, 800, 200, 80, 60, 20, 1.8, 1.0]
    y = [5.9, 5.4, 4.4, 4.6, 3.5, 3.7, 2.8, 2.8, 2.4, 1.5]
    wy = [1.0, 1.8, 4.0, 8.0, 20.0, 20.0, 70.0, 70.0, 100.0, 500.0]
    ux = [1. / math.sqrt(wx_i) for wx_i in wx]
    uy = [1. / math.sqrt(wy_i) for wy_i in wy]

    gtc = type_a.line_fit_wtls(x, y, ux, uy)

    with LinearModel() as model:
        model.options(weighted=True, y_uncertainties_only=False)
        params = model.create_parameters([('a1', 5, False, 'intercept'),
                                          ('a2', -1, False, 'slope')])
        nlf = model.fit(x=x, y=y, params=params, uy=uy, ux=ux)

    assert approx(gtc.intercept.x, abs=0.0002) == nlf.params['intercept'].value
    assert approx(gtc.intercept.u, abs=0.02) == nlf.params['intercept'].uncert
    assert approx(gtc.slope.x, abs=0.002) == nlf.params['slope'].value
    assert approx(gtc.slope.u, abs=0.02) == nlf.params['slope'].uncert
    assert approx(gtc.ssr, abs=0.002) == nlf.chisq


def test_to_ureal_1():
    # Only 1 parameter in model

    x = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    y = [10.1, 5.2, 2.1, 1.3, 0.9, 0.5, 0.4, 0.1, 0.05, 0.08]

    with Model('a1*exp(-x)') as model:
        result = model.fit(x, y, params=[1])

    # the NLF GUI was used to get the right-hand side values
    assert approx(result.chisq) == 3.80528448730178
    assert result.correlation.shape == (1, 1)
    assert approx(result.correlation[0, 0]) == 1.0
    assert result.covariance.shape == (1, 1)
    assert approx(result.covariance[0, 0]) == 0.86466472
    assert result.dof == 9
    assert approx(result.eof) == 0.650237946814838
    values = result.params.values()
    assert values.shape == (1,)
    assert approx(values[0]) == 10.7070245442441
    uncerts = result.params.uncerts()
    assert uncerts.shape == (1,)
    assert approx(uncerts[0]) == 0.604639032830398

    ensemble = result.to_ureal()
    assert len(ensemble) == 1
    a1 = ensemble[0]
    assert approx(a1.x) == 10.7070245442441
    assert approx(a1.u) == 0.604639032830398
    assert get_correlation(a1, a1) == 1.0

    ensemble = result.to_ureal(with_future=True)
    assert len(ensemble) == 2
    a1, noise = ensemble
    assert approx(a1.x) == 10.7070245442441
    assert approx(a1.u) == 0.604639032830398
    assert get_correlation(a1, a1) == 1.0
    assert approx(noise.x) == 0.0
    assert approx(noise.u) == math.sqrt(result.chisq/result.dof)
    assert get_correlation(a1, noise) == 0.0


def test_to_ureal_2():
    # 2 parameters in model
    # x, y values were taken from:
    #   https://gtc.readthedocs.io/en/stable/API/linear_regression.html

    x = [3, 7, 11, 15, 18, 27, 29, 30, 30, 31, 31, 32, 33, 33, 34, 36,
         36, 36, 37, 38, 39, 39, 39, 40, 41, 42, 42, 43, 44, 45, 46, 47, 50]
    y = [5, 11, 21, 16, 16, 28, 27, 25, 35, 30, 40, 32, 34, 32, 34, 37,
         38, 34, 36, 38, 37, 36, 45, 39, 41, 40, 44, 37, 44, 46, 46, 49, 51]

    gtc = type_a.line_fit(x, y)

    with LinearModel() as model:
        result = model.fit(x, y, params=[1, 1])

    intercept, slope = result.to_ureal()
    assert approx(intercept.x) == gtc.intercept.x
    assert approx(intercept.u) == gtc.intercept.u
    assert intercept.df == gtc.intercept.df
    assert approx(slope.x) == gtc.slope.x
    assert approx(slope.u) == gtc.slope.u
    assert slope.df == gtc.slope.df
    assert approx(get_correlation(intercept, slope)) == get_correlation(gtc.intercept, gtc.slope)

    # The response (see GTC docs)
    y_nlf = intercept + 21.5 * slope
    y_gtc = gtc.intercept + 21.5 * gtc.slope
    assert approx(y_nlf.x) == y_gtc.x
    assert approx(y_nlf.u) == y_gtc.u
    assert y_nlf.df == y_gtc.df

    # A predicted future response (see GTC docs)
    intercept, slope, noise = result.to_ureal(with_future=True)
    y_nlf = intercept + 21.5 * slope + noise
    y_gtc = gtc.y_from_x(21.5)
    assert approx(y_nlf.x) == y_gtc.x
    assert approx(y_nlf.u) == y_gtc.u
    assert y_nlf.df == y_gtc.df


def test_to_ureal_6():
    # 6 parameters in model
    # do not perform a fit, create a Result object

    # only the upper triangle is used
    anan = np.nan
    correlation = np.array([[1.00, 0.12, 0.13, 0.14, 0.15, 0.16],
                            [anan, 1.00, 0.23, 0.24, 0.25, 0.26],
                            [anan, anan, 1.00, 0.34, 0.35, 0.36],
                            [anan, anan, anan, 1.00, 0.45, 0.46],
                            [anan, anan, anan, anan, 1.00, 0.56],
                            [anan, anan, anan, anan, anan, 1.00]])

    params = ResultParameters(
        {'a': [1, 2, 3, 4, 5, 6],
         'ua': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]},
        InputParameters((
            ('a1', -1, False, 'label1'),
            ('a2', -2, False, 'label2'),
            ('a3', -3, False, 'label3'),
            ('a4', -4, False, 'label4'),
            ('a5', -5, False, 'label5'),
            ('a6', -6, False, 'label6')))
    )

    kwargs = {
        'calls': 2,
        'chisq': 11.1,
        'correlation': correlation,
        'covariance': np.arange(36).reshape((6, 6)),
        'dof': 12,
        'eof': 0.321,
        'iterations': 20,
        'params': params,
    }

    ensemble = Result(**kwargs).to_ureal()
    assert len(ensemble) == 6

    a1, a2, a3, a4, a5, a6 = ensemble
    assert a1.x == 1.0
    assert a1.u == 0.1
    assert a1.df == 12
    assert a1.label == 'label1'
    assert a2.x == 2.0
    assert a2.u == 0.2
    assert a2.df == 12
    assert a2.label == 'label2'
    assert a3.x == 3.0
    assert a3.u == 0.3
    assert a3.df == 12
    assert a3.label == 'label3'
    assert a4.x == 4.0
    assert a4.u == 0.4
    assert a4.df == 12
    assert a4.label == 'label4'
    assert a5.x == 5.0
    assert a5.u == 0.5
    assert a5.df == 12
    assert a5.label == 'label5'
    assert a6.x == 6.0
    assert a6.u == 0.6
    assert a6.df == 12
    assert a6.label == 'label6'

    assert get_correlation(a1, a1) == 1.00
    assert get_correlation(a1, a2) == 0.12
    assert get_correlation(a1, a3) == 0.13
    assert get_correlation(a1, a4) == 0.14
    assert get_correlation(a1, a5) == 0.15
    assert get_correlation(a1, a6) == 0.16

    assert get_correlation(a2, a1) == 0.12
    assert get_correlation(a2, a2) == 1.00
    assert get_correlation(a2, a3) == 0.23
    assert get_correlation(a2, a4) == 0.24
    assert get_correlation(a2, a5) == 0.25
    assert get_correlation(a2, a6) == 0.26

    assert get_correlation(a3, a1) == 0.13
    assert get_correlation(a3, a2) == 0.23
    assert get_correlation(a3, a3) == 1.00
    assert get_correlation(a3, a4) == 0.34
    assert get_correlation(a3, a5) == 0.35
    assert get_correlation(a3, a6) == 0.36

    assert get_correlation(a4, a1) == 0.14
    assert get_correlation(a4, a2) == 0.24
    assert get_correlation(a4, a3) == 0.34
    assert get_correlation(a4, a4) == 1.00
    assert get_correlation(a4, a5) == 0.45
    assert get_correlation(a4, a6) == 0.46

    assert get_correlation(a5, a1) == 0.15
    assert get_correlation(a5, a2) == 0.25
    assert get_correlation(a5, a3) == 0.35
    assert get_correlation(a5, a4) == 0.45
    assert get_correlation(a5, a5) == 1.00
    assert get_correlation(a5, a6) == 0.56

    assert get_correlation(a6, a1) == 0.16
    assert get_correlation(a6, a2) == 0.26
    assert get_correlation(a6, a3) == 0.36
    assert get_correlation(a6, a4) == 0.46
    assert get_correlation(a6, a5) == 0.56
    assert get_correlation(a6, a6) == 1.00

    ensemble = Result(**kwargs).to_ureal(with_future=True, label='noise')
    assert len(ensemble) == 7

    a1, a2, a3, a4, a5, a6, noise = ensemble
    assert a1.x == 1.0
    assert a1.u == 0.1
    assert a1.df == 12
    assert a1.label == 'label1'
    assert a2.x == 2.0
    assert a2.u == 0.2
    assert a2.df == 12
    assert a2.label == 'label2'
    assert a3.x == 3.0
    assert a3.u == 0.3
    assert a3.df == 12
    assert a3.label == 'label3'
    assert a4.x == 4.0
    assert a4.u == 0.4
    assert a4.df == 12
    assert a4.label == 'label4'
    assert a5.x == 5.0
    assert a5.u == 0.5
    assert a5.df == 12
    assert a5.label == 'label5'
    assert a6.x == 6.0
    assert a6.u == 0.6
    assert a6.df == 12
    assert a6.label == 'label6'
    assert noise.x == 0.0
    assert approx(noise.u) == math.sqrt(kwargs['chisq']/float(kwargs['dof']))
    assert noise.df == 12
    assert noise.label == 'noise'

    assert get_correlation(a1, a1) == 1.00
    assert get_correlation(a1, a2) == 0.12
    assert get_correlation(a1, a3) == 0.13
    assert get_correlation(a1, a4) == 0.14
    assert get_correlation(a1, a5) == 0.15
    assert get_correlation(a1, a6) == 0.16
    assert get_correlation(a1, noise) == 0.00

    assert get_correlation(a2, a1) == 0.12
    assert get_correlation(a2, a2) == 1.00
    assert get_correlation(a2, a3) == 0.23
    assert get_correlation(a2, a4) == 0.24
    assert get_correlation(a2, a5) == 0.25
    assert get_correlation(a2, a6) == 0.26
    assert get_correlation(a2, noise) == 0.00

    assert get_correlation(a3, a1) == 0.13
    assert get_correlation(a3, a2) == 0.23
    assert get_correlation(a3, a3) == 1.00
    assert get_correlation(a3, a4) == 0.34
    assert get_correlation(a3, a5) == 0.35
    assert get_correlation(a3, a6) == 0.36
    assert get_correlation(a3, noise) == 0.00

    assert get_correlation(a4, a1) == 0.14
    assert get_correlation(a4, a2) == 0.24
    assert get_correlation(a4, a3) == 0.34
    assert get_correlation(a4, a4) == 1.00
    assert get_correlation(a4, a5) == 0.45
    assert get_correlation(a4, a6) == 0.46
    assert get_correlation(a4, noise) == 0.00

    assert get_correlation(a5, a1) == 0.15
    assert get_correlation(a5, a2) == 0.25
    assert get_correlation(a5, a3) == 0.35
    assert get_correlation(a5, a4) == 0.45
    assert get_correlation(a5, a5) == 1.00
    assert get_correlation(a5, a6) == 0.56
    assert get_correlation(a5, noise) == 0.00

    assert get_correlation(a6, a1) == 0.16
    assert get_correlation(a6, a2) == 0.26
    assert get_correlation(a6, a3) == 0.36
    assert get_correlation(a6, a4) == 0.46
    assert get_correlation(a6, a5) == 0.56
    assert get_correlation(a6, a6) == 1.00
    assert get_correlation(a6, noise) == 0.00

    assert get_correlation(noise, a1) == 0.00
    assert get_correlation(noise, a2) == 0.00
    assert get_correlation(noise, a3) == 0.00
    assert get_correlation(noise, a4) == 0.00
    assert get_correlation(noise, a5) == 0.00
    assert get_correlation(noise, a6) == 0.00
    assert get_correlation(noise, noise) == 1.00
