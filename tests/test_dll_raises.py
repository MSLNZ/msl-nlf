import pytest
from msl.loadlib import Server32Error

from msl.nlf import FitMethod
from msl.nlf import LinearModel
from msl.nlf import Model
from msl.nlf.model import IS_PYTHON_64BIT


def test_runtime_error():
    with LinearModel(weighted=True) as model:
        with pytest.raises(RuntimeError, match=r'The uncertainties are not complete'):
            model.fit(x=[1, 2, 3, 4], y=[1, 2, 3, 4], params=[1, 1])


@pytest.mark.skipif(not IS_PYTHON_64BIT, reason='32-bit Python, expect RuntimeError not Server32Error')
def test_server32_error():
    with LinearModel(weighted=True, dll='nlf32') as model:
        with pytest.raises(Server32Error, match=r'The uncertainties are not complete'):
            model.fit(x=[1, 2, 3, 4], y=[1, 2, 3, 4], params=[1, 1])


@pytest.mark.parametrize(
    'method',
    [FitMethod.AMOEBA_MD,
     FitMethod.POWELL_MD,
     FitMethod.AMOEBA_MM,
     FitMethod.POWELL_MM])
def test_correlated(method):
    # Minimum Distance and MiniMax are invalid for correlated data
    match = r'cannot be performed as a correlated fit'
    with LinearModel(fit_method=method, correlated=True) as model:
        model.set_correlation('x', 'y', value=0.5)
        with pytest.raises(RuntimeError, match=match):
            model.fit(x=[1, 2, 3], y=[1, 2, 3], params=[1, 1])


@pytest.mark.parametrize('method', [FitMethod.AMOEBA_MD, FitMethod.POWELL_MD])
def test_minimum_distance_nvars(method):
    # Minimum Distance is invalid if nvars > 1
    match = r'can only be performed for functions with one x-variable'
    with Model('a1*x + a2*x2', fit_method=method) as model:
        with pytest.raises(RuntimeError, match=match):
            model.fit(x=[[1, 2, 3], [1, 2, 3]], y=[1, 2, 3], params=[1, 1])
