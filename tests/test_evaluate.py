import numpy as np
import pytest

from msl.nlf import *
from msl.nlf.datatypes import Result
from msl.nlf.parameter import InputParameters
from msl.nlf.parameter import ResultParameters


def test_sin():
    a1 = 3.4
    a2 = 10.3
    a3 = 0.6

    def sin(t):
        return a1 * np.sin(a2 * t + a3)

    # pass in a name:value mapping
    x = np.linspace(0, 1, 100)
    with SineModel() as model:
        y = model.evaluate(x, {'a1': a1, 'a2': a2, 'a3': a3})
    assert np.allclose(sin(x), y)

    result = Result(
        calls=1,
        chisq=1.0,
        correlation=np.zeros(10),
        covariance=np.zeros(10),
        dof=1,
        eof=1.0,
        iterations=1,
        params=ResultParameters(
            {'a': [a1, a2, a3], 'ua': [0.1, 0.1, 0.1], 'dof': 1},
            InputParameters([('a1', a1), ('a2', a2), ('a3', a3)])
        ))

    with SineModel() as model:
        y2 = model.evaluate(x, result)

    # pass in a Result object
    assert np.allclose(sin(x), y2)


def test_cos():
    a1 = 6.1
    a2 = 9.5
    a3 = -1.4

    def cos(t):
        return a1 * np.cos(a2 * t + a3)

    x = np.linspace(0, 1, 100)
    with CosineModel() as model:
        y = model.evaluate(x, {'a1': a1, 'a2': a2, 'a3': a3})

    assert np.allclose(cos(x), y)


def test_linear():
    a1 = 0.2
    a2 = 9.5

    def linear(t):
        return a1 + a2*t

    x = np.linspace(0, 1, 100)
    with LinearModel() as model:
        y = model.evaluate(x, {'a1': a1, 'a2': a2})

    assert np.allclose(linear(x), y)


def test_polynomial():
    a1 = 0.2
    a2 = 9.5
    a3 = 0.2
    a4 = 1e-2
    a5 = 2.1e-3
    a6 = 5.3e-4
    a7 = 8.7e-8

    def poly(t):
        return a1 + a2*t + a3*t**2 + a4*t**3 + a5*t**4 + a6*t**5 + a7*t**6

    x = np.linspace(0, 1, 100)
    with PolynomialModel(6) as model:
        y = model.evaluate(x, {'a1': a1, 'a2': a2, 'a3': a3, 'a4': a4,
                               'a5': a5, 'a6': a6, 'a7': a7})

    assert np.allclose(poly(x), y)


def test_exponential():
    a1 = 11.1
    a2 = 2.3

    def exponential(t):
        return a1 * np.exp(-a2 * t)

    x = np.linspace(0, 1, 100)
    with ExponentialModel() as model:
        y = model.evaluate(x, {'a1': a1, 'a2': a2})

    assert np.allclose(exponential(x), y)


def test_tan():
    a1 = 4.2
    a2 = 7.3
    a3 = -2.4

    def tan(t):
        return a1 * np.tan(a2 * t + a3)

    x = np.linspace(0, 1, 100)
    with Model('a1*tan(a2*x+a3)') as model:
        y = model.evaluate(x, {'a1': a1, 'a2': a2, 'a3': a3})

    assert np.allclose(tan(x), y)


def test_ln():
    a1 = 4.2
    a2 = 7.3

    def ln(t):
        return a1 + np.log(t / a2)

    x = np.linspace(1, 2, 100)
    with Model('a1+ln(x/a2)') as model:
        y = model.evaluate(x, {'a1': a1, 'a2': a2})

    assert np.allclose(ln(x), y)


def test_log():
    a1 = 2.9
    a2 = 6.1

    def log(t):
        return a1 + np.log10(t / a2)

    x = np.linspace(1, 2, 100)
    with Model('a1+log(x/a2)') as model:
        y = model.evaluate(x, {'a1': a1, 'a2': a2})

    assert np.allclose(log(x), y)


def test_arcsin():
    a1 = 4.2
    a2 = 0.97

    def arcsin(t):
        return a1 * np.arcsin(a2 * t)

    x = np.linspace(-1, 1, 100)
    with Model('a1*arcsin(a2*x)') as model:
        y = model.evaluate(x, {'a1': a1, 'a2': a2})

    assert np.allclose(arcsin(x), y)


def test_arcos():
    a1 = 4.2
    a2 = 0.97

    def arcos(t):
        return a1 * np.arccos(a2 * t)

    x = np.linspace(-1, 1, 100)
    with Model('a1*arcos(a2*x)') as model:
        y = model.evaluate(x, {'a1': a1, 'a2': a2})

    assert np.allclose(arcos(x), y)


def test_ugly():
    a1 = 4.2
    a2 = 0.97
    a3 = 8.6
    a4 = 0.7

    def ugly(t):
        return (a1 * np.arccos(a2 * t) + 273.15 - a4*x**3) / \
            (2*np.sin(a3*t) - 8*np.exp(-a4*t) + np.log(a3)*t)

    x = np.linspace(-1, 1, 100)
    equation = '(a1*arcos(a2*x)+273.15-a4*x^3)/(2*sin(a3*x)-8*exp(-a4*x)+ln(a3)*x)'
    with Model(equation) as model:
        y = model.evaluate(x, {'a1': a1, 'a2': a2, 'a3': a3, 'a4': a4})

    assert np.allclose(ugly(x), y)


@pytest.mark.parametrize(
    'equation',
    ['a1+a2*(x+exp(a3*x))+x2',     # use x instead of x1
     'a1+a2*(x1+exp(a3*x1))+x2'])  # use x1 and x2
def test_multiple_variables(equation):
    x = np.array([[1, 2, 3, 4], [0.1, 0.2, 0.3, 0.4]])
    y = np.array([1.1, 1.9, 3.2, 3.7])
    guess = np.array([0, 0.9, 0])

    model = Model(equation)
    result = model.fit(x, y, params=guess)

    with pytest.raises(ValueError, match=r'Invalid shape of x data'):
        model.evaluate([[[1, 2], [3, 4], [5, 6]]], result)

    with pytest.raises(ValueError, match=r'Unexpected number of x variables'):
        model.evaluate([1, 2, 3, 4], result)

    with pytest.raises(ValueError, match=r'Unexpected number of x variables'):
        model.evaluate([[1, 2], [3, 4], [5, 6]], result)

    y_fit = model.evaluate(x, result)
    assert pytest.approx(result.chisq) == np.sum((y - y_fit)**2)
