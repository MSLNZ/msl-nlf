import os

import numpy as np
import pytest

from msl.nlf import Model
from msl.nlf.dll import get_user_defined
from msl.nlf.model import IS_PYTHON_64BIT

if IS_PYTHON_64BIT:
    dlls = ['nlf64', 'nlf32']
else:
    dlls = ['nlf32']


@pytest.mark.parametrize(
    'equation, name',
    [('f1', 'f1: Roszman1 f1=a1-a2*x-arctan(a3/(x-a4))/pi'),
     ('f2', 'f2: Nelson log(f2)=a1-a2*x1*exp(-a3*x2)')])
def test_valid(equation, name):
    with Model(equation, user_dir='./tests/user_defined') as model:
        assert model.equation == equation
        assert model.user_function_name == name


@pytest.mark.parametrize('equation', ['f 1', 'f1 0', 'f1:'])
def test_invalid(equation):
    # invalid but no exception is raised until a fit is performed
    with Model(equation) as model:
        assert model.equation == equation
        with pytest.raises(RuntimeError, match=r'Invalid Equation'):
            model.fit([1, 2, 3], [1, 2, 3], params=[])


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
    assert len(functions) == 2
    for ud in functions.values():
        if ud.equation == 'f1':
            assert ud.name == 'f1: Roszman1 f1=a1-a2*x-arctan(a3/(x-a4))/pi'
            assert ud.function is not None
            assert ud.num_parameters == 4
            assert ud.num_variables == 1
        elif ud.equation == 'f2':
            assert ud.name == 'f2: Nelson log(f2)=a1-a2*x1*exp(-a3*x2)'
            assert ud.function is not None
            assert ud.num_parameters == 3
            assert ud.num_variables == 2
        else:
            raise ValueError('Unexpected equation value')


@pytest.mark.parametrize('dll', dlls)
def test_roszman1(dll):
    # See NIST_datasets/Roszman1.dat for the numerical values

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


@pytest.mark.parametrize('dll', dlls)
def test_nelson(dll):
    # See NIST_datasets/Nelson.dat for the numerical values

    x = [[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
          1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0,
          2.0, 2.0, 2.0, 2.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0,
          4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0,
          8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 16.0, 16.0, 16.0, 16.0, 16.0,
          16.0, 16.0, 16.0, 16.0, 16.0, 16.0, 16.0, 16.0, 16.0, 16.0, 16.0, 32.0,
          32.0, 32.0, 32.0, 32.0, 32.0, 32.0, 32.0, 32.0, 32.0, 32.0, 32.0, 32.0,
          32.0, 32.0, 32.0, 48.0, 48.0, 48.0, 48.0, 48.0, 48.0, 48.0, 48.0, 48.0,
          48.0, 48.0, 48.0, 48.0, 48.0, 48.0, 48.0, 64.0, 64.0, 64.0, 64.0, 64.0,
          64.0, 64.0, 64.0, 64.0, 64.0, 64.0, 64.0, 64.0, 64.0, 64.0, 64.0],
         [180.0, 180.0, 180.0, 180.0, 225.0, 225.0, 225.0, 225.0, 250.0, 250.0,
          250.0, 250.0, 275.0, 275.0, 275.0, 275.0, 180.0, 180.0, 180.0, 180.0,
          225.0, 225.0, 225.0, 225.0, 250.0, 250.0, 250.0, 250.0, 275.0, 275.0,
          275.0, 275.0, 180.0, 180.0, 180.0, 180.0, 225.0, 225.0, 225.0, 225.0,
          250.0, 250.0, 250.0, 250.0, 275.0, 275.0, 275.0, 275.0, 180.0, 180.0,
          180.0, 180.0, 225.0, 225.0, 225.0, 225.0, 250.0, 250.0, 250.0, 250.0,
          275.0, 275.0, 275.0, 275.0, 180.0, 180.0, 180.0, 180.0, 225.0, 225.0,
          225.0, 225.0, 250.0, 250.0, 250.0, 250.0, 275.0, 275.0, 275.0, 275.0,
          180.0, 180.0, 180.0, 180.0, 225.0, 225.0, 225.0, 225.0, 250.0, 250.0,
          250.0, 250.0, 275.0, 275.0, 275.0, 275.0, 180.0, 180.0, 180.0, 180.0,
          225.0, 225.0, 225.0, 225.0, 250.0, 250.0, 250.0, 250.0, 275.0, 275.0,
          275.0, 275.0, 180.0, 180.0, 180.0, 180.0, 225.0, 225.0, 225.0, 225.0,
          250.0, 250.0, 250.0, 250.0, 275.0, 275.0, 275.0, 275.0]]

    y = np.log([15.0, 17.0, 15.5, 16.5, 15.5, 15.0, 16.0, 14.5, 15.0, 14.5,
                12.5, 11.0, 14.0, 13.0, 14.0, 11.5, 14.0, 16.0, 13.0, 13.5,
                13.0, 13.5, 12.5, 12.5, 12.5, 12.0, 11.5, 12.0, 13.0, 11.5,
                13.0, 12.5, 13.5, 17.5, 17.5, 13.5, 12.5, 12.5, 15.0, 13.0,
                12.0, 13.0, 12.0, 13.5, 10.0, 11.5, 11.0, 9.5, 15.0, 15.0,
                15.5, 16.0, 13.0, 10.5, 13.5, 14.0, 12.5, 12.0, 11.5, 11.5,
                6.5, 5.5, 6.0, 6.0, 18.5, 17.0, 15.3, 16.0, 13.0, 14.0, 12.5,
                11.0, 12.0, 12.0, 11.5, 12.0, 6.0, 6.0, 5.0, 5.5, 12.5, 13.0,
                16.0, 12.0, 11.0, 9.5, 11.0, 11.0, 11.0, 10.0, 10.5, 10.5, 2.7,
                2.7, 2.5, 2.4, 13.0, 13.5, 16.5, 13.6, 11.5, 10.5, 13.5, 12.0,
                7.0, 6.9, 8.8, 7.9, 1.2, 1.5, 1.0, 1.5, 13.0, 12.5, 16.5, 16.0,
                11.0, 11.5, 10.5, 10.0, 7.27, 7.5, 6.7, 7.6, 1.5, 1.0, 1.2, 1.2])

    params = [2.0, 0.0001, -0.01]

    chisq_expected = 3.7976833176E+00
    eof_expected = 1.7430280130E-01

    with Model('f2', dll=dll, user_dir='./tests/user_defined') as model:
        assert model.equation == 'f2'
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


@pytest.mark.parametrize(
    'user_dir',
    ['./tests/user_defined',
     './tests/user_defined/',
     '.\\tests\\user_defined',
     '.\\tests\\user_defined\\',
     os.path.join(os.path.dirname(__file__), 'user_defined'),
     os.path.dirname(__file__) + '/user_defined',
     os.path.dirname(__file__) + '/user_defined/',
     os.path.dirname(__file__) + r'\user_defined',
     os.path.dirname(__file__) + '\\user_defined\\',
     ])
def test_path_seperator_ending(user_dir):
    with Model('f1', user_dir=user_dir) as model:
        assert model.equation == 'f1'
        assert model.user_function_name == 'f1: Roszman1 f1=a1-a2*x-arctan(a3/(x-a4))/pi'
