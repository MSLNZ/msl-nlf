import numpy as np
import pytest
from msl.loadlib import IS_PYTHON_64BIT

from msl.nlf import Model
from msl.nlf.dll import get_user_defined

if IS_PYTHON_64BIT:
    dlls = ['nlf64', 'nlf32']
else:
    dlls = ['nlf32']


def test_valid():
    with Model('f1', user_dir='./tests/user_defined') as model:
        assert model.equation == 'f1'


@pytest.mark.parametrize('equation', ['f 1', 'f1 0', 'f1:'])
def test_invalid(equation):
    # invalid but no exception is raised until a fit is performed
    with Model(equation) as model:
        assert model.equation == equation
        with pytest.raises(ValueError, match=r'Invalid equation'):
            model.fit([1, 2, 3], [1, 2, 3])


@pytest.mark.parametrize('dll', dlls)
def test_does_not_exist(dll):
    with pytest.raises(ValueError, match=r'No user-defined function'):
        with Model('f4', dll=dll, user_dir='./tests/user_defined'):
            pass


@pytest.mark.parametrize('dll', dlls)
def test_multiple_exist(dll):
    with pytest.raises(ValueError, match=r'Multiple user-defined functions'):
        with Model('f1', dll=dll, user_dir='./tests/user_defined/multiple'):
            pass


@pytest.mark.parametrize('dll', dlls)
def test_none_exist(dll):
    with pytest.raises(ValueError, match=r'no valid functions'):
        with Model('f1', dll=dll, user_dir='./tests/user_defined/only_invalid'):
            pass


def test_invalid_directory():
    with pytest.raises(FileNotFoundError):
        with Model('f1', user_dir='invalid'):
            pass


def test_get_user_defined():
    assert get_user_defined('.') == {}
    assert get_user_defined('./tests') == {}

    functions = get_user_defined('./tests/user_defined')
    assert len(functions) == 1
    for ud in functions.values():
        if ud.equation == 'f1':
            assert ud.name == 'f1: Roszman1 f1=a1-a2*x-arctan(a3/(x-a4))/pi'
            assert ud.function is not None
            assert ud.num_parameters == 4
            assert ud.num_variables == 1
        else:
            raise ValueError('Unexpected equation value')


@pytest.mark.parametrize('dll', dlls)
def test_roszman1(dll):
    x = [-4868.68, -4868.09, -4867.41, -3375.19, -3373.14, -3372.03,
         -2473.74, -2472.35, -2469.45, -1894.65, -1893.40, -1497.24,
         -1495.85, -1493.41, -1208.68, -1206.18, -1206.04, -997.92,
         -996.61, -996.31, -834.94, -834.66, -710.03, -530.16, -464.17]

    y = np.array([
        0.252429, 0.252141, 0.251809, 0.297989, 0.296257, 0.295319,
        0.339603, 0.337731, 0.333820, 0.389510, 0.386998, 0.438864,
        0.434887, 0.427893, 0.471568, 0.461699, 0.461144, 0.513532,
        0.506641, 0.505062, 0.535648, 0.533726, 0.568064, 0.612886,
        0.624169])

    params = [0.1, -1e-5, 1e3, -1e2]

    # See NIST_datasets/Roszman1.dat for the numerical values
    chisq_expected = 4.9484847331E-04
    eof_expected = 4.8542984060E-03

    with Model('f1', dll=dll, user_dir='./tests/user_defined') as model:
        assert model.equation == 'f1'
        result = model.fit(x, y, params=params)

        # use the Result object
        residuals = y - model.evaluate(x, result)
        chisq = np.sum(np.square(residuals))
        eof = np.sqrt(chisq / (len(y) - len(params)))
        assert pytest.approx(chisq_expected) == chisq
        assert pytest.approx(eof_expected) == eof

        # use a mapping
        r = dict((p.name, p.value) for p in result.params)
        residuals = y - model.evaluate(x, r)
        chisq = np.sum(np.square(residuals))
        eof = np.sqrt(chisq / (len(y) - len(params)))
        assert pytest.approx(chisq_expected) == chisq
        assert pytest.approx(eof_expected) == eof
