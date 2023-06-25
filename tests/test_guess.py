import math

import numpy as np
import pytest

from msl.nlf import *
from msl.nlf.models import _mean_max_n  # noqa
from msl.nlf.models import _mean_min_n  # noqa


def test_mean_max_n():
    a = np.arange(1234)
    np.random.shuffle(a)
    assert _mean_max_n(a, 1) == 1233.
    assert _mean_max_n(a, 2) == sum([1232, 1233]) / 2.0
    assert _mean_max_n(a, 10) == sum([1224, 1225, 1226, 1227, 1228,
                                      1229, 1230, 1231, 1232, 1233])/10.0


def test_mean_min_n():
    a = np.arange(1234)
    np.random.shuffle(a)
    assert _mean_min_n(a, 1) == 0.0
    assert _mean_min_n(a, 2) == sum([0, 1]) / 2.0
    assert _mean_min_n(a, 10) == sum([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]) / 10.0


def test_not_implemented():
    with pytest.raises(NotImplementedError):
        with Model('a1*x') as model:
            model.guess([1, 2], [3, 4])


def test_constant():
    y = np.array([0.691818, 0.887067, 0.048726, 0.010144, 0.336313,
                  0.396944, 0.118929, 0.795026, 0.864225, 0.811257])

    with ConstantModel() as model:
        params = model.guess([], y)  # x is not used
        c = params['constant']
        assert c.name == 'a1'
        assert pytest.approx(c.value) == 0.4960449
        assert c.constant is False
        assert c.label == 'constant'

        params = model.guess([], y, n=3)  # x is not used
        c = params['constant']
        assert c.name == 'a1'
        assert pytest.approx(c.value) == 0.542537
        assert c.constant is False
        assert c.label == 'constant'

        params = model.guess([], y, n=-2)  # x is not used
        c = params['constant']
        assert c.name == 'a1'
        assert pytest.approx(c.value) == 0.837741
        assert c.constant is False
        assert c.label == 'constant'


def test_linear():
    x = np.array([1.6, 3.2, 5.5, 7.8, 9.4])
    y = np.array([7.8, 19.1, 17.6, 33.9, 45.4])

    with LinearModel() as model:
        params = model.guess(x, y)

    intercept = params['intercept']
    assert intercept.name == 'a1'
    assert pytest.approx(intercept.value) == 0.522439024390246
    assert intercept.constant is False
    assert intercept.label == 'intercept'

    slope = params['slope']
    assert slope.name == 'a2'
    assert pytest.approx(slope.value) == 4.40682926829268
    assert slope.constant is False
    assert slope.label == 'slope'


def test_linear_fit_default_guess():
    # do not specify params to model.fit() and the model.guess()
    # method is called automatically

    x = np.array([1.6, 3.2, 5.5, 7.8, 9.4])
    y = np.array([7.8, 19.1, 17.6, 33.9, 45.4])

    with LinearModel() as model:
        result = model.fit(x, y)

    intercept = result.params['intercept']
    slope = result.params['slope']
    assert pytest.approx(intercept.value) == 0.5224390243902463
    assert pytest.approx(intercept.uncert) == 5.132418149940029
    assert pytest.approx(slope.value) == 4.406829268292682
    assert pytest.approx(slope.uncert) == 0.8277017245089597


@pytest.mark.parametrize(
    'n, names, labels',
    [(1, ['a1', 'a2'], ['a1', 'a2*x']),
     (2, ['a1', 'a2', 'a3'], ['a1', 'a2*x', 'a3*x^2']),
     (3, ['a1', 'a2', 'a3', 'a4'], ['a1', 'a2*x', 'a3*x^2', 'a4*x^3']),
     (4, ['a1', 'a2', 'a3', 'a4', 'a5'], ['a1', 'a2*x', 'a3*x^2', 'a4*x^3', 'a5*x^4'])])
def test_polynomial(n, names, labels):
    x = np.array([1.6, 3.2, 5.5, 7.8, 9.4])
    y = np.array([7.8, 19.1, 17.6, 33.9, 45.4])

    with PolynomialModel(n) as model:
        params = model.guess(x, y)

    # don't care about the values, since that is handle by numpy
    assert len(params) == n + 1
    assert params.names() == names
    assert np.array_equal(params.constants(), [False] * (n + 1))
    assert params.labels() == labels


@pytest.mark.parametrize(
    'amp, decay',
    [(1.2, 0.4),
     (3.2, 1.1),
     (3.2, -1.1),
     (1e3, 3.3),
     (100.7, 2.9)])
def test_exponential(amp, decay):
    x = np.linspace(0, 10)
    y = amp * np.exp(-decay * x)
    with ExponentialModel() as model:
        params = model.guess(x, y)
        assert len(params) == 2
        assert pytest.approx(amp, rel=1e-4) == params['amplitude'].value
        assert pytest.approx(decay, rel=1e-4) == params['decay'].value


@pytest.mark.parametrize(
    'amp, decay',
    [(1.2, 0.4),
     (3.2, 1.1),
     (1e3, 3.3),
     (100.7, 2.9)])
def test_exponential_cumulative(amp, decay):
    x = np.linspace(0, 10)
    y = amp * (1.0 - np.exp(-decay * x))
    with ExponentialModel(cumulative=True) as model:
        params = model.guess(x, y)
        assert len(params) == 2
        assert pytest.approx(amp, abs=0.025) == params['amplitude'].value
        assert pytest.approx(decay, abs=0.25) == params['decay'].value


@pytest.mark.parametrize(
    'amplitude, mu, sigma',
    [(123, 3.3, 0.8),
     (123, -3.3, 0.8),
     (123, -3.3, 2.8),
     (-123, -3.3, 2.8),
     (-123, -9.3, 1.0),
     (1, 9.3, 0.5),
     (10, 0, 10)])
def test_gaussian(amplitude, mu, sigma):
    x = np.linspace(-10, 10)
    y = amplitude * np.exp(-0.5 * ((x-mu)/sigma)**2)
    with GaussianModel() as model:
        params = model.guess(x, y)
        assert len(params) == 3
        assert pytest.approx(amplitude) == params['amplitude'].value
        assert pytest.approx(mu) == params['mu'].value
        assert pytest.approx(sigma) == params['sigma'].value


@pytest.mark.parametrize(
    'area, mu, sigma',
    [(123, 3.3, 0.8),
     (123, -3.3, 0.8),
     (123, -3.3, 2.8),
     (-123, -3.3, 2.8),
     (-123, -9.3, 1.0),
     (1, 9.3, 0.5),
     (10, 0, 10)])
def test_gaussian_normalized(area, mu, sigma):
    x = np.linspace(-10, 10)
    y = area / (sigma * math.sqrt(2.0 * math.pi)) * np.exp(-0.5 * ((x-mu)/sigma)**2)
    with GaussianModel(normalized=True) as model:
        params = model.guess(x, y)
        assert len(params) == 3
        assert pytest.approx(area) == params['area'].value
        assert pytest.approx(mu) == params['mu'].value
        assert pytest.approx(sigma) == params['sigma'].value


@pytest.mark.parametrize(
    'amplitude, omega, phase',
    [(10, 25, 0.23),
     (1, 50, 3.86)])
def test_sin(amplitude, omega, phase):
    x = np.linspace(-1, 1, 100)
    y = amplitude * np.sin(omega*x + phase)
    with SineModel() as model:
        params = model.guess(x, y)
        assert len(params) == 3
        assert pytest.approx(amplitude, rel=1e-2) == params['amplitude'].value
        assert pytest.approx(omega, rel=1e-2) == params['omega'].value
        assert pytest.approx(phase, abs=2*math.pi/11) == params['phase'].value
